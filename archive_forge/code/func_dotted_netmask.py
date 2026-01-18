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
def dotted_netmask(mask):
    """Converts mask from /xx format to xxx.xxx.xxx.xxx

    Example: if mask is 24 function returns 255.255.255.0

    :rtype: str
    """
    bits = 4294967295 ^ (1 << 32 - mask) - 1
    return socket.inet_ntoa(struct.pack('>I', bits))