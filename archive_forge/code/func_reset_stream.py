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
def reset_stream(self, reason: StreamCloseReason=StreamCloseReason.RESET) -> None:
    """Close this stream by sending a RST_FRAME to the remote peer"""
    if self.metadata['stream_closed_local']:
        raise StreamClosedError(self.stream_id)
    self._response['body'].truncate(0)
    self.metadata['stream_closed_local'] = True
    self._protocol.conn.reset_stream(self.stream_id, ErrorCodes.REFUSED_STREAM)
    self.close(reason)