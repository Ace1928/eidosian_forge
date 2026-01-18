from __future__ import annotations
from typing import List, Optional
from typing_extensions import Literal
import httpx
from .... import _legacy_response
from .files import (
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ...._utils import (
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from .file_batches import (
from ....pagination import SyncCursorPage, AsyncCursorPage
from ....types.beta import (
from ...._base_client import (
@cached_property
def file_batches(self) -> AsyncFileBatchesWithStreamingResponse:
    return AsyncFileBatchesWithStreamingResponse(self._vector_stores.file_batches)