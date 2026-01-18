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
def initiate_request(self) -> None:
    if self.check_request_url():
        headers = self._get_request_headers()
        self._protocol.conn.send_headers(self.stream_id, headers, end_stream=False)
        self.metadata['request_sent'] = True
        self.send_data()
    else:
        self.close(StreamCloseReason.INVALID_HOSTNAME)