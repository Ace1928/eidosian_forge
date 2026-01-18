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
def push_session(document: Document, session_id: ID | None=None, url: str='default', io_loop: IOLoop | None=None, max_message_size: int=20 * 1024 * 1024) -> ClientSession:
    """ Create a session by pushing the given document to the server,
    overwriting any existing server-side document.

    ``session.document`` in the returned session will be your supplied
    document. While the connection to the server is open, changes made on the
    server side will be applied to this document, and changes made on the
    client side will be synced to the server.

    In a production scenario, the ``session_id`` should be unique for each
    browser tab, which keeps users from stomping on each other. It's neither
    scalable nor secure to use predictable session IDs or to share session
    IDs across users.

    For a notebook running on a single machine, ``session_id`` could be
    something human-readable such as ``"default"`` for convenience.

    If you allow ``push_session()`` to generate a unique ``session_id``, you
    can obtain the generated ID with the ``id`` property on the returned
    ``ClientSession``.

    Args:
        document : (bokeh.document.Document)
            The document to be pushed and set as session.document

        session_id : (string, optional)
            The name of the session, None to autogenerate a random one (default: None)

        url : (str, optional): The URL to a Bokeh application on a Bokeh server
            can also be `"default"` which will connect to the default app URL

        io_loop : (tornado.ioloop.IOLoop, optional)
            The IOLoop to use for the websocket

        max_message_size (int, optional) :
            Configure the Tornado max websocket message size.
            (default: 20 MB)

    Returns:
        ClientSession
            A new ClientSession connected to the server

    """
    coords = SessionCoordinates(session_id=session_id, url=url)
    session = ClientSession(session_id=coords.session_id, websocket_url=websocket_url_for_server_url(coords.url), io_loop=io_loop, max_message_size=max_message_size)
    session.push(document)
    return session