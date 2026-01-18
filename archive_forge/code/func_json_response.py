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
def json_response(data: Any=sentinel, *, text: Optional[str]=None, body: Optional[bytes]=None, status: int=200, reason: Optional[str]=None, headers: Optional[LooseHeaders]=None, content_type: str='application/json', dumps: JSONEncoder=json.dumps) -> Response:
    if data is not sentinel:
        if text or body:
            raise ValueError('only one of data, text, or body should be specified')
        else:
            text = dumps(data)
    return Response(text=text, body=body, status=status, reason=reason, headers=headers, content_type=content_type)