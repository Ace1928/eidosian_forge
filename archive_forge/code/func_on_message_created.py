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
def on_message_created(self, message: Message) -> None:
    """Callback that is fired when a message is created"""