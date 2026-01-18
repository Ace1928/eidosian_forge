import asyncio
import asyncio.streams
import traceback
import warnings
from collections import deque
from contextlib import suppress
from html import escape as html_escape
from http import HTTPStatus
from logging import Logger
from typing import (
import attr
import yarl
from .abc import AbstractAccessLogger, AbstractStreamWriter
from .base_protocol import BaseProtocol
from .helpers import ceil_timeout
from .http import (
from .log import access_logger, server_logger
from .streams import EMPTY_PAYLOAD, StreamReader
from .tcp_helpers import tcp_keepalive
from .web_exceptions import HTTPException
from .web_log import AccessLogger
from .web_request import BaseRequest
from .web_response import Response, StreamResponse
@property
def keepalive_timeout(self) -> float:
    return self._keepalive_timeout