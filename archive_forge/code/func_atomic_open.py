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
@contextlib.contextmanager
def atomic_open(filename):
    """Write a file to the disk in an atomic fashion"""
    replacer = os.rename if sys.version_info[0] == 2 else os.replace
    tmp_descriptor, tmp_name = tempfile.mkstemp(dir=os.path.dirname(filename))
    try:
        with os.fdopen(tmp_descriptor, 'wb') as tmp_handler:
            yield tmp_handler
        replacer(tmp_name, filename)
    except BaseException:
        os.remove(tmp_name)
        raise