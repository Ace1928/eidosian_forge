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
def guess_json_utf(data):
    """
    :rtype: str
    """
    sample = data[:4]
    if sample in (codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE):
        return 'utf-32'
    if sample[:3] == codecs.BOM_UTF8:
        return 'utf-8-sig'
    if sample[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return 'utf-16'
    nullcount = sample.count(_null)
    if nullcount == 0:
        return 'utf-8'
    if nullcount == 2:
        if sample[::2] == _null2:
            return 'utf-16-be'
        if sample[1::2] == _null2:
            return 'utf-16-le'
    if nullcount == 3:
        if sample[:3] == _null3:
            return 'utf-32-be'
        if sample[1:] == _null3:
            return 'utf-32-le'
    return None