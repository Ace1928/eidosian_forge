from __future__ import (absolute_import, division, print_function)
import atexit
import base64
import email.mime.multipart
import email.mime.nonmultipart
import email.mime.application
import email.parser
import email.utils
import functools
import io
import mimetypes
import netrc
import os
import platform
import re
import socket
import sys
import tempfile
import traceback
import types  # pylint: disable=unused-import
from contextlib import contextmanager
import ansible.module_utils.compat.typing as t
import ansible.module_utils.six.moves.http_cookiejar as cookiejar
import ansible.module_utils.six.moves.urllib.error as urllib_error
from ansible.module_utils.common.collections import Mapping, is_sequence
from ansible.module_utils.six import PY2, PY3, string_types
from ansible.module_utils.six.moves import cStringIO
from ansible.module_utils.basic import get_distribution, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def prepare_multipart(fields):
    """Takes a mapping, and prepares a multipart/form-data body

    :arg fields: Mapping
    :returns: tuple of (content_type, body) where ``content_type`` is
        the ``multipart/form-data`` ``Content-Type`` header including
        ``boundary`` and ``body`` is the prepared bytestring body

    Payload content from a file will be base64 encoded and will include
    the appropriate ``Content-Transfer-Encoding`` and ``Content-Type``
    headers.

    Example:
        {
            "file1": {
                "filename": "/bin/true",
                "mime_type": "application/octet-stream"
            },
            "file2": {
                "content": "text based file content",
                "filename": "fake.txt",
                "mime_type": "text/plain",
            },
            "text_form_field": "value"
        }
    """
    if not isinstance(fields, Mapping):
        raise TypeError('Mapping is required, cannot be type %s' % fields.__class__.__name__)
    m = email.mime.multipart.MIMEMultipart('form-data')
    for field, value in sorted(fields.items()):
        if isinstance(value, string_types):
            main_type = 'text'
            sub_type = 'plain'
            content = value
            filename = None
        elif isinstance(value, Mapping):
            filename = value.get('filename')
            content = value.get('content')
            if not any((filename, content)):
                raise ValueError('at least one of filename or content must be provided')
            mime = value.get('mime_type')
            if not mime:
                try:
                    mime = mimetypes.guess_type(filename or '', strict=False)[0] or 'application/octet-stream'
                except Exception:
                    mime = 'application/octet-stream'
            main_type, sep, sub_type = mime.partition('/')
        else:
            raise TypeError('value must be a string, or mapping, cannot be type %s' % value.__class__.__name__)
        if not content and filename:
            with open(to_bytes(filename, errors='surrogate_or_strict'), 'rb') as f:
                part = email.mime.application.MIMEApplication(f.read())
                del part['Content-Type']
                part.add_header('Content-Type', '%s/%s' % (main_type, sub_type))
        else:
            part = email.mime.nonmultipart.MIMENonMultipart(main_type, sub_type)
            part.set_payload(to_bytes(content))
        part.add_header('Content-Disposition', 'form-data')
        del part['MIME-Version']
        part.set_param('name', field, header='Content-Disposition')
        if filename:
            part.set_param('filename', to_native(os.path.basename(filename)), header='Content-Disposition')
        m.attach(part)
    if PY3:
        b_data = m.as_bytes(policy=email.policy.HTTP)
    else:
        fp = cStringIO()
        g = email.generator.Generator(fp, maxheaderlen=0)
        g.flatten(m)
        b_data = email.utils.fix_eols(fp.getvalue())
    del m
    headers, sep, b_content = b_data.partition(b'\r\n\r\n')
    del b_data
    if PY3:
        parser = email.parser.BytesHeaderParser().parsebytes
    else:
        parser = email.parser.HeaderParser().parsestr
    return (parser(headers)['content-type'], b_content)