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
@property
def body_file_seekable(self):
    """
            Get the body of the request (wsgi.input) as a seekable file-like
            object. Middleware and routing applications should use this
            attribute over .body_file.

            If you access this value, CONTENT_LENGTH will also be updated.
        """
    if not self.is_body_seekable:
        self.make_body_seekable()
    return self.body_file_raw