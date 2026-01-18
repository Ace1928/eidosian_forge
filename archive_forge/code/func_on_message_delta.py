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
def on_message_delta(self, delta: MessageDelta, snapshot: Message) -> None:
    """Callback that is fired whenever a message delta is returned from the API

        The first argument is just the delta as sent by the API and the second argument
        is the accumulated snapshot of the message. For example, a text content event may
        look like this:

        # delta
        MessageDeltaText(
            index=0,
            type='text',
            text=Text(
                value=' Jane'
            ),
        )
        # snapshot
        MessageContentText(
            index=0,
            type='text',
            text=Text(
                value='Certainly, Jane'
            ),
        )
        """