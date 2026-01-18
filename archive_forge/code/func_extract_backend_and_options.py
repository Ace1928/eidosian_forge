from __future__ import annotations
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import Any, Dict, Tuple, cast
import pytest
import sniffio
from ._core._eventloop import get_all_backends, get_async_backend
from .abc import TestRunner
def extract_backend_and_options(backend: object) -> tuple[str, dict[str, Any]]:
    if isinstance(backend, str):
        return (backend, {})
    elif isinstance(backend, tuple) and len(backend) == 2:
        if isinstance(backend[0], str) and isinstance(backend[1], dict):
            return cast(Tuple[str, Dict[str, Any]], backend)
    raise TypeError('anyio_backend must be either a string or tuple of (string, dict)')