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
@property
def async_pipeline(self) -> AsyncPipeline:
    """
        Initializes a `AsyncKeyDB` Client and returns a `AsyncPipeline`.
        """
    if self.state.async_pipeline is None:
        self.state.async_pipeline = self.async_client.pipeline(transaction=self.config.get('transaction', True), shard_hint=self.config.get('shard_hint', None))
    return self.state.async_pipeline