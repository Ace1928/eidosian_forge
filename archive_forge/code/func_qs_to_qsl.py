import calendar
import collections.abc
import copy
import datetime
import email.utils
from functools import lru_cache
from http.client import responses
import http.cookies
import re
from ssl import SSLError
import time
import unicodedata
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from tornado.escape import native_str, parse_qs_bytes, utf8
from tornado.log import gen_log
from tornado.util import ObjectDict, unicode_type
import typing
from typing import (
def qs_to_qsl(qs: Dict[str, List[AnyStr]]) -> Iterable[Tuple[str, AnyStr]]:
    """Generator converting a result of ``parse_qs`` back to name-value pairs.

    .. versionadded:: 5.0
    """
    for k, vs in qs.items():
        for v in vs:
            yield (k, v)