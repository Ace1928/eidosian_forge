from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote_plus
from ..core.types import ID
from ..document import Document
from ..resources import DEFAULT_SERVER_HTTP_URL, SessionCoordinates
from ..util.browser import NEW_PARAM, BrowserLike, BrowserTarget
from ..util.token import generate_jwt_token, generate_session_id
from .states import ErrorReason
from .util import server_url_for_websocket_url, websocket_url_for_server_url
def force_roundtrip(self) -> None:
    """ Force a round-trip request/reply to the server, sometimes needed to
        avoid race conditions. Mostly useful for testing.

        Outside of test suites, this method hurts performance and should not be
        needed.

        Returns:
           None

        """
    self._connection.force_roundtrip()