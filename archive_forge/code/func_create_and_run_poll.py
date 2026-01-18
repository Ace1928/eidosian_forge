from __future__ import annotations
from typing import Union, Iterable, Optional, overload
from functools import partial
from typing_extensions import Literal
import httpx
from .... import _legacy_response
from .runs import (
from .messages import (
from ...._types import NOT_GIVEN, Body, Query, Headers, NotGiven
from ...._utils import (
from .runs.runs import Runs, AsyncRuns
from ...._compat import cached_property
from ...._resource import SyncAPIResource, AsyncAPIResource
from ...._response import to_streamed_response_wrapper, async_to_streamed_response_wrapper
from ...._streaming import Stream, AsyncStream
from ....types.beta import (
from ...._base_client import (
from ....lib.streaming import (
from ....types.beta.threads import Run
def create_and_run_poll(self, *, assistant_id: str, instructions: Optional[str] | NotGiven=NOT_GIVEN, max_completion_tokens: Optional[int] | NotGiven=NOT_GIVEN, max_prompt_tokens: Optional[int] | NotGiven=NOT_GIVEN, metadata: Optional[object] | NotGiven=NOT_GIVEN, model: Union[str, Literal['gpt-4-turbo', 'gpt-4-turbo-2024-04-09', 'gpt-4-0125-preview', 'gpt-4-turbo-preview', 'gpt-4-1106-preview', 'gpt-4-vision-preview', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0314', 'gpt-4-32k-0613', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo-0125', 'gpt-3.5-turbo-16k-0613'], None] | NotGiven=NOT_GIVEN, response_format: Optional[AssistantResponseFormatOptionParam] | NotGiven=NOT_GIVEN, temperature: Optional[float] | NotGiven=NOT_GIVEN, thread: thread_create_and_run_params.Thread | NotGiven=NOT_GIVEN, tool_choice: Optional[AssistantToolChoiceOptionParam] | NotGiven=NOT_GIVEN, tool_resources: Optional[thread_create_and_run_params.ToolResources] | NotGiven=NOT_GIVEN, tools: Optional[Iterable[thread_create_and_run_params.Tool]] | NotGiven=NOT_GIVEN, top_p: Optional[float] | NotGiven=NOT_GIVEN, truncation_strategy: Optional[thread_create_and_run_params.TruncationStrategy] | NotGiven=NOT_GIVEN, poll_interval_ms: int | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
    """
        A helper to create a thread, start a run and then poll for a terminal state.
        More information on Run lifecycles can be found here:
        https://platform.openai.com/docs/assistants/how-it-works/runs-and-run-steps
        """
    run = self.create_and_run(assistant_id=assistant_id, instructions=instructions, max_completion_tokens=max_completion_tokens, max_prompt_tokens=max_prompt_tokens, metadata=metadata, model=model, response_format=response_format, temperature=temperature, stream=False, thread=thread, tool_resources=tool_resources, tool_choice=tool_choice, truncation_strategy=truncation_strategy, top_p=top_p, tools=tools, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout)
    return self.runs.poll(run.id, run.thread_id, extra_headers, extra_query, extra_body, timeout, poll_interval_ms)