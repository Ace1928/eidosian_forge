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
def check_connection_errors(self) -> None:
    """ Raises an error, when the connection could not have been
        established.

        Should be used, after a call to connect.

        Returns:
            None

        """
    if not self.connected:
        if self.error_reason is ErrorReason.HTTP_ERROR:
            if self.error_code == 404:
                raise OSError(f'Check your application path! The given Path is not valid: {self.url}')
            raise OSError(f'We received an HTTP-Error. Disconnected with error code: {self.error_code}, given message: {self.error_detail}')
        raise OSError("We failed to connect to the server (to start the server, try the 'bokeh serve' command)")