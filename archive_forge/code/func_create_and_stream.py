from __future__ import annotations
from typing import Iterable, Optional, overload
from functools import partial
from typing_extensions import Literal
import httpx
from ..... import _legacy_response
from .steps import (
from ....._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ....._utils import (
from ....._compat import cached_property
from ....._resource import SyncAPIResource, AsyncAPIResource
from ....._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ....._streaming import Stream, AsyncStream
from .....pagination import SyncCursorPage, AsyncCursorPage
from .....types.beta import AssistantToolParam, AssistantStreamEvent
from ....._base_client import (
from .....lib.streaming import (
from .....types.beta.threads import (
def create_and_stream(self, *, assistant_id: str, additional_instructions: Optional[str] | NotGiven=NOT_GIVEN, instructions: Optional[str] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, model: Optional[str] | NotGiven=NOT_GIVEN, tools: Optional[Iterable[AssistantToolParam]] | NotGiven=NOT_GIVEN, thread_id: str, event_handler: AsyncAssistantEventHandlerT | None=None, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> AsyncAssistantStreamManager[AsyncAssistantEventHandler] | AsyncAssistantStreamManager[AsyncAssistantEventHandlerT]:
    """Create a Run stream"""
    if not thread_id:
        raise ValueError(f'Expected a non-empty value for `thread_id` but received {thread_id!r}')
    extra_headers = {'OpenAI-Beta': 'assistants=v1', 'X-Stainless-Stream-Helper': 'threads.runs.create_and_stream', 'X-Stainless-Custom-Event-Handler': 'true' if event_handler else 'false', **(extra_headers or {})}
    request = self._post(f'/threads/{thread_id}/runs', body=maybe_transform({'assistant_id': assistant_id, 'additional_instructions': additional_instructions, 'instructions': instructions, 'metadata': metadata, 'model': model, 'stream': True, 'tools': tools}, run_create_params.RunCreateParams), options=make_request_options(extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout), cast_to=Run, stream=True, stream_cls=AsyncStream[AssistantStreamEvent])
    return AsyncAssistantStreamManager(request, event_handler=event_handler or AsyncAssistantEventHandler())