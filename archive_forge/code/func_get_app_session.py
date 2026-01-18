from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Generator
def get_app_session() -> AppSession:
    return _current_app_session.get()