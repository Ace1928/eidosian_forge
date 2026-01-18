import sys
import time
import anyio
import typing
import logging
import asyncio
import functools
import contextlib
from pydantic import BaseModel
from pydantic.types import ByteSize
from aiokeydb.v2.typing import Number, KeyT, ExpiryT, AbsExpiryT, PatternT
from aiokeydb.v2.lock import Lock, AsyncLock
from aiokeydb.v2.core import KeyDB, PubSub, Pipeline, PipelineT, PubSubT
from aiokeydb.v2.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline, AsyncPipelineT, AsyncPubSubT
from aiokeydb.v2.connection import Encoder, ConnectionPool, AsyncConnectionPool
from aiokeydb.v2.exceptions import (
from aiokeydb.v2.types import KeyDBUri, ENOVAL
from aiokeydb.v2.configs import KeyDBSettings, settings as default_settings
from aiokeydb.v2.utils import full_name, args_to_key
from aiokeydb.v2.utils.helpers import create_retryable_client
from aiokeydb.v2.serializers import BaseSerializer
from inspect import iscoroutinefunction
def plisten(self, *patterns: str, timeout: typing.Optional[float]=None, unsubscribe_after: typing.Optional[bool]=False, close_after: typing.Optional[bool]=False, listen_callback: typing.Optional[typing.Callable]=None, cancel_callback: typing.Optional[typing.Callable]=None, **kwargs) -> typing.Iterator[typing.Any]:
    """
        [PubSub] Listens for messages
        """

    def _listen():
        if timeout:
            from lazyops.utils import fail_after
            with contextlib.suppress(TimeoutError):
                with fail_after(timeout):
                    for message in self.pubsub.listen():
                        if listen_callback:
                            listen_callback(message, **kwargs)
                        if cancel_callback and cancel_callback(message, **kwargs):
                            break
                        yield message
        else:
            for message in self.pubsub.listen():
                if listen_callback:
                    listen_callback(message, **kwargs)
                if cancel_callback and cancel_callback(message, **kwargs):
                    break
                yield message
    try:
        if patterns:
            self.pubsub.psubscribe(*patterns)
        yield from _listen()
    finally:
        if unsubscribe_after:
            self.pubsub.unsubscribe(*patterns)
        if close_after:
            self.pubsub.close()