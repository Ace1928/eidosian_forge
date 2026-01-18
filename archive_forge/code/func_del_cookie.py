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
def del_cookie(self, name: str, *, domain: Optional[str]=None, path: str='/') -> None:
    """Delete cookie.

        Creates new empty expired cookie.
        """
    self._cookies.pop(name, None)
    self.set_cookie(name, '', max_age=0, expires='Thu, 01 Jan 1970 00:00:00 GMT', domain=domain, path=path)