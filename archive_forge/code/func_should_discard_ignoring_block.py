from __future__ import annotations
import logging # isort:skip
import weakref
from typing import (
from tornado import gen
from ..application.application import ServerContext, SessionContext
from ..document import Document
from ..protocol.exceptions import ProtocolError
from ..util.token import get_token_payload
from .session import ServerSession
def should_discard_ignoring_block(session: ServerSession) -> bool:
    return session.connection_count == 0 and (session.milliseconds_since_last_unsubscribe > unused_session_linger_milliseconds or session.expiration_requested)