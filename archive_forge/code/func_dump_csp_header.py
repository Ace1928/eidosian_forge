from __future__ import annotations
import email.utils
import re
import typing as t
import warnings
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from enum import Enum
from hashlib import sha1
from time import mktime
from time import struct_time
from urllib.parse import quote
from urllib.parse import unquote
from urllib.request import parse_http_list as _parse_list_header
from ._internal import _dt_as_utc
from ._internal import _plain_int
from . import datastructures as ds
from .sansio import http as _sansio_http
def dump_csp_header(header: ds.ContentSecurityPolicy) -> str:
    """Dump a Content Security Policy header.

    These are structured into policies such as "default-src 'self';
    script-src 'self'".

    .. versionadded:: 1.0.0
       Support for Content Security Policy headers was added.

    """
    return '; '.join((f'{key} {value}' for key, value in header.items()))