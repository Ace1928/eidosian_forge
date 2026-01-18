import datetime
import sys
import encodings.idna
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata
from urllib3.util import parse_url
from urllib3.exceptions import (
from io import UnsupportedOperation
from .hooks import default_hooks
from .structures import CaseInsensitiveDict
from .auth import HTTPBasicAuth
from .cookies import cookiejar_from_dict, get_cookie_header, _copy_cookie_jar
from .exceptions import (
from .exceptions import JSONDecodeError as RequestsJSONDecodeError
from .exceptions import SSLError as RequestsSSLError
from ._internal_utils import to_native_string, unicode_is_ascii
from .utils import (
from .compat import (
from .compat import json as complexjson
from .status_codes import codes
def prepare_body(self, data, files, json=None):
    """Prepares the given HTTP body data."""
    body = None
    content_type = None
    if not data and json is not None:
        content_type = 'application/json'
        try:
            body = complexjson.dumps(json, allow_nan=False)
        except ValueError as ve:
            raise InvalidJSONError(ve, request=self)
        if not isinstance(body, bytes):
            body = body.encode('utf-8')
    is_stream = all([hasattr(data, '__iter__'), not isinstance(data, (basestring, list, tuple, Mapping))])
    if is_stream:
        try:
            length = super_len(data)
        except (TypeError, AttributeError, UnsupportedOperation):
            length = None
        body = data
        if getattr(body, 'tell', None) is not None:
            try:
                self._body_position = body.tell()
            except (IOError, OSError):
                self._body_position = object()
        if files:
            raise NotImplementedError('Streamed bodies and files are mutually exclusive.')
        if length:
            self.headers['Content-Length'] = builtin_str(length)
        else:
            self.headers['Transfer-Encoding'] = 'chunked'
    else:
        if files:
            body, content_type = self._encode_files(files, data)
        elif data:
            body = self._encode_params(data)
            if isinstance(data, basestring) or hasattr(data, 'read'):
                content_type = None
            else:
                content_type = 'application/x-www-form-urlencoded'
        self.prepare_content_length(body)
        if content_type and 'content-type' not in self.headers:
            self.headers['Content-Type'] = content_type
    self.body = body