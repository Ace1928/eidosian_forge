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
def do_discard() -> None:
    if should_discard(session) and session.expiration_blocked_count == 1:
        session.destroy()
        del self._sessions[session.id]
        del self._session_contexts[session.id]
        log.trace(f'Session {session.id!r} was successfully discarded')
    else:
        log.warning(f'Session {session.id!r} was scheduled to discard but came back to life')