from __future__ import annotations
import time
import typing_extensions
from typing import Union, Iterable, Optional, overload
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
from .....types.beta import (
from ....._base_client import (
from .....lib.streaming import (
from .....types.beta.threads import (
def submit_tool_outputs_and_poll(self, *, tool_outputs: Iterable[run_submit_tool_outputs_params.ToolOutput], run_id: str, thread_id: str, poll_interval_ms: int | NotGiven=NOT_GIVEN, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN) -> Run:
    """
        A helper to submit a tool output to a run and poll for a terminal run state.
        More information on Run lifecycles can be found here:
        https://platform.openai.com/docs/assistants/how-it-works/runs-and-run-steps
        """
    run = self.submit_tool_outputs(run_id=run_id, thread_id=thread_id, tool_outputs=tool_outputs, stream=False, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout)
    return self.poll(run_id=run.id, thread_id=thread_id, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout, poll_interval_ms=poll_interval_ms)