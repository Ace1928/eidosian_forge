from __future__ import annotations
import os
import inspect
import logging
import datetime
import functools
from typing import TYPE_CHECKING, Any, Union, Generic, TypeVar, Callable, Iterator, AsyncIterator, cast, overload
from typing_extensions import Awaitable, ParamSpec, override, deprecated, get_origin
import anyio
import httpx
import pydantic
from ._types import NoneType
from ._utils import is_given, extract_type_arg, is_annotated_type
from ._models import BaseModel, is_basemodel
from ._constants import RAW_RESPONSE_HEADER
from ._streaming import Stream, AsyncStream, is_stream_class_type, extract_stream_chunk_type
from ._exceptions import APIResponseValidationError
@deprecated("Due to a bug, this method doesn't actually stream the response content, `.with_streaming_response.method()` should be used instead")
def stream_to_file(self, file: str | os.PathLike[str], *, chunk_size: int | None=None) -> None:
    with open(file, mode='wb') as f:
        for data in self.response.iter_bytes(chunk_size):
            f.write(data)