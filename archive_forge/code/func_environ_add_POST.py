import binascii
import io
import os
import re
import sys
import tempfile
import mimetypes
import warnings
from webob.acceptparse import (
from webob.cachecontrol import (
from webob.compat import (
from webob.cookies import RequestCookies
from webob.descriptors import (
from webob.etag import (
from webob.headers import EnvironHeaders
from webob.multidict import (
def environ_add_POST(env, data, content_type=None):
    if data is None:
        return
    elif isinstance(data, text_type):
        data = data.encode('ascii')
    if env['REQUEST_METHOD'] not in ('POST', 'PUT'):
        env['REQUEST_METHOD'] = 'POST'
    has_files = False
    if hasattr(data, 'items'):
        data = list(data.items())
        for k, v in data:
            if isinstance(v, (tuple, list)):
                has_files = True
                break
    if content_type is None:
        if has_files:
            content_type = 'multipart/form-data'
        else:
            content_type = 'application/x-www-form-urlencoded'
    if content_type.startswith('multipart/form-data'):
        if not isinstance(data, bytes):
            content_type, data = _encode_multipart(data, content_type)
    elif content_type.startswith('application/x-www-form-urlencoded'):
        if has_files:
            raise ValueError('Submiting files is not allowed for content type `%s`' % content_type)
        if not isinstance(data, bytes):
            data = url_encode(data)
    elif not isinstance(data, bytes):
        raise ValueError('Please provide `POST` data as bytes for content type `%s`' % content_type)
    data = bytes_(data, 'utf8')
    env['wsgi.input'] = io.BytesIO(data)
    env['webob.is_body_seekable'] = True
    env['CONTENT_LENGTH'] = str(len(data))
    env['CONTENT_TYPE'] = content_type