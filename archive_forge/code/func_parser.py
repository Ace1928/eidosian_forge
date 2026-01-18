from __future__ import annotations
import base64
from typing import List, Union, Iterable, cast
from typing_extensions import Literal
import httpx
from .. import _legacy_response
from ..types import CreateEmbeddingResponse, embedding_create_params
from .._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from .._utils import is_given, maybe_transform
from .._compat import cached_property
from .._extras import numpy as np, has_numpy
from .._resource import SyncAPIResource, AsyncAPIResource
from .._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .._base_client import (
def parser(obj: CreateEmbeddingResponse) -> CreateEmbeddingResponse:
    if is_given(encoding_format):
        return obj
    for embedding in obj.data:
        data = cast(object, embedding.embedding)
        if not isinstance(data, str):
            continue
        embedding.embedding = np.frombuffer(base64.b64decode(data), dtype='float32').tolist()
    return obj