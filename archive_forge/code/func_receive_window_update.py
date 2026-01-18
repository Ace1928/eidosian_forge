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
def receive_window_update(self) -> None:
    """Flow control window size was changed.
        Send data that earlier could not be sent as we were
        blocked behind the flow control.
        """
    if self.metadata['remaining_content_length'] and (not self.metadata['stream_closed_server']) and self.metadata['request_sent']:
        self.send_data()