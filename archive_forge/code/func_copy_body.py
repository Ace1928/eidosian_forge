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
def copy_body(self):
    """
        Copies the body, in cases where it might be shared with another request
        object and that is not desired.

        This copies the body either into a BytesIO object (through setting
        req.body) or a temporary file.
        """
    if self.is_body_readable:
        if self.is_body_seekable:
            self.body_file_raw.seek(0)
        tempfile_limit = self.request_body_tempfile_limit
        todo = self.content_length if self.content_length is not None else 65535
        newbody = b''
        fileobj = None
        input = self.body_file
        while todo > 0:
            data = input.read(min(todo, 65535))
            if not data and self.content_length is None:
                break
            elif not data:
                raise DisconnectionError('Client disconnected (%s more bytes were expected)' % todo)
            if fileobj:
                fileobj.write(data)
            else:
                newbody += data
                if len(newbody) > tempfile_limit:
                    fileobj = self.make_tempfile()
                    fileobj.write(newbody)
                    newbody = b''
            if self.content_length is not None:
                todo -= len(data)
        if fileobj:
            self.content_length = fileobj.tell()
            fileobj.seek(0)
            self.body_file_raw = fileobj
            self.is_body_seekable = True
            self.is_body_readable = True
        else:
            self.body = newbody
    else:
        self.body = b''