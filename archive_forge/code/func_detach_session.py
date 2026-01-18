from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, Awaitable
def detach_session(self) -> None:
    """Allow the session to be discarded and don't get change notifications from it anymore"""
    if self._session is not None:
        self._session.unsubscribe(self)
        self._session = None