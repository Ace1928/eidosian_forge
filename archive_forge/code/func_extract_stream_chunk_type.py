from __future__ import annotations
import json
import inspect
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Iterator, AsyncIterator, cast
from typing_extensions import Self, Protocol, TypeGuard, override, get_origin, runtime_checkable
import httpx
from ._utils import is_mapping, extract_type_var_from_base
from ._exceptions import APIError
def extract_stream_chunk_type(stream_cls: type, *, failure_message: str | None=None) -> type:
    """Given a type like `Stream[T]`, returns the generic type variable `T`.

    This also handles the case where a concrete subclass is given, e.g.
    ```py
    class MyStream(Stream[bytes]):
        ...

    extract_stream_chunk_type(MyStream) -> bytes
    ```
    """
    from ._base_client import Stream, AsyncStream
    return extract_type_var_from_base(stream_cls, index=0, generic_bases=cast('tuple[type, ...]', (Stream, AsyncStream)), failure_message=failure_message)