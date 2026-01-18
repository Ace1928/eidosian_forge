from __future__ import annotations
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from inspect import isasyncgenfunction, iscoroutinefunction
from typing import Any, Dict, Tuple, cast
import pytest
import sniffio
from ._core._eventloop import get_all_backends, get_async_backend
from .abc import TestRunner
@pytest.fixture
def anyio_backend_options(anyio_backend: Any) -> dict[str, Any]:
    if isinstance(anyio_backend, str):
        return {}
    else:
        return anyio_backend[1]