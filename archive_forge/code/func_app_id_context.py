from __future__ import annotations
import sys
from contextlib import contextmanager
from contextvars import ContextVar
@contextmanager
def app_id_context(app_id: str):
    token = _app_id_context_var.set(app_id)
    yield
    _app_id_context_var.reset(token)