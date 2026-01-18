import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict
from urllib3.util import make_headers
from urllib3.util import parse_url
from .__version__ import __version__
from . import certs
from ._internal_utils import to_native_string
from .compat import parse_http_list as _parse_list_header
from .compat import (
from .cookies import cookiejar_from_dict
from .structures import CaseInsensitiveDict
from .exceptions import (
def rewind_body(prepared_request):
    """Move file pointer back to its recorded starting position
    so it can be read again on redirect.
    """
    body_seek = getattr(prepared_request.body, 'seek', None)
    if body_seek is not None and isinstance(prepared_request._body_position, integer_types):
        try:
            body_seek(prepared_request._body_position)
        except (IOError, OSError):
            raise UnrewindableBodyError('An error occurred when rewinding request body for redirect.')
    else:
        raise UnrewindableBodyError('Unable to rewind request body for redirect.')