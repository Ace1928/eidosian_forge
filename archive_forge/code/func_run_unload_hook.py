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
def run_unload_hook(self) -> None:
    try:
        self._application.on_server_unloaded(self.server_context)
    except Exception as e:
        log.error(f'Error in server unloaded hook {e!r}', exc_info=True)