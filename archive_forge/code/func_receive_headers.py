import logging
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from h2.errors import ErrorCodes
from h2.exceptions import H2Error, ProtocolError, StreamClosedError
from hpack import HeaderTuple
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.error import ConnectionClosed
from twisted.python.failure import Failure
from twisted.web.client import ResponseFailed
from scrapy.http import Request
from scrapy.http.headers import Headers
from scrapy.responsetypes import responsetypes
def receive_headers(self, headers: List[HeaderTuple]) -> None:
    for name, value in headers:
        self._response['headers'].appendlist(name, value)
    expected_size = int(self._response['headers'].get(b'Content-Length', -1))
    if self._download_maxsize and expected_size > self._download_maxsize:
        self.reset_stream(StreamCloseReason.MAXSIZE_EXCEEDED)
        return
    if self._log_warnsize:
        self.metadata['reached_warnsize'] = True
        warning_msg = f'Expected response size ({expected_size}) larger than download warn size ({self._download_warnsize}) in request {self._request}'
        logger.warning(warning_msg)