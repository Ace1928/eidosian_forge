from __future__ import annotations
import json
import inspect
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Iterator, AsyncIterator, cast
from typing_extensions import Self, Protocol, TypeGuard, override, get_origin, runtime_checkable
import httpx
from ._utils import is_mapping, extract_type_var_from_base
from ._exceptions import APIError
def aiter_bytes(self, iterator: AsyncIterator[bytes]) -> AsyncIterator[ServerSentEvent]:
    """Given an async iterator that yields raw binary data, iterate over it & yield every event encountered"""
    ...