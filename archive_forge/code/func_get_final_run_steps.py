from __future__ import annotations
import asyncio
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Callable, Iterable, Iterator, cast
from typing_extensions import Awaitable, AsyncIterable, AsyncIterator, assert_never
import httpx
from ..._utils import is_dict, is_list, consume_sync_iterator, consume_async_iterator
from ..._models import construct_type
from ..._streaming import Stream, AsyncStream
from ...types.beta import AssistantStreamEvent
from ...types.beta.threads import (
from ...types.beta.threads.runs import RunStep, ToolCall, RunStepDelta, ToolCallDelta
def get_final_run_steps(self) -> list[RunStep]:
    """Wait for the stream to finish and returns the steps taken in this run"""
    self.until_done()
    if not self.__run_step_snapshots:
        raise RuntimeError('No run steps found')
    return [step for step in self.__run_step_snapshots.values()]