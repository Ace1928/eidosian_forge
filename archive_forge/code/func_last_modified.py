import asyncio
import collections.abc
import datetime
import enum
import json
import math
import time
import warnings
from concurrent.futures import Executor
from http import HTTPStatus
from http.cookies import SimpleCookie
from typing import (
from multidict import CIMultiDict, istr
from . import hdrs, payload
from .abc import AbstractStreamWriter
from .compression_utils import ZLibCompressor
from .helpers import (
from .http import SERVER_SOFTWARE, HttpVersion10, HttpVersion11
from .payload import Payload
from .typedefs import JSONEncoder, LooseHeaders
@last_modified.setter
def last_modified(self, value: Optional[Union[int, float, datetime.datetime, str]]) -> None:
    if value is None:
        self._headers.pop(hdrs.LAST_MODIFIED, None)
    elif isinstance(value, (int, float)):
        self._headers[hdrs.LAST_MODIFIED] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime(math.ceil(value)))
    elif isinstance(value, datetime.datetime):
        self._headers[hdrs.LAST_MODIFIED] = time.strftime('%a, %d %b %Y %H:%M:%S GMT', value.utctimetuple())
    elif isinstance(value, str):
        self._headers[hdrs.LAST_MODIFIED] = value