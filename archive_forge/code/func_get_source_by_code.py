from __future__ import annotations
import code
import sys
import typing as t
from contextvars import ContextVar
from types import CodeType
from markupsafe import escape
from .repr import debug_repr
from .repr import dump
from .repr import helper
def get_source_by_code(self, code: CodeType) -> str | None:
    try:
        return self._storage[id(code)]
    except KeyError:
        return None