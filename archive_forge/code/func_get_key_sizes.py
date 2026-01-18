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
def get_key_sizes(self, match: typing.Union[PatternT, None]=None, count: typing.Union[int, None]=None, _type: typing.Union[str, None]=None, min_size: typing.Union[ByteSize, int, str, None]=None, max_size: typing.Union[ByteSize, int, str, None]=None, raise_error: typing.Optional[bool]=False, parse: typing.Optional[bool]=True, verbose: typing.Optional[bool]=True, **kwargs) -> typing.Iterator[typing.Tuple[str, typing.Union[ByteSize, int]]]:
    """
        Returns an iterator that yields a tuple of key name and size in bytes or a ByteSize object
        """
    if min_size is not None and (not isinstance(min_size, ByteSize)):
        min_size = ByteSize.validate(min_size)
    if max_size is not None and (not isinstance(max_size, ByteSize)):
        max_size = ByteSize.validate(max_size)
    for key in self.scan_iter(match=match, count=count, _type=_type, **kwargs):
        try:
            size = self.strlen(key)
            if parse:
                size = ByteSize.validate(size)
            if min_size is not None and size < min_size:
                continue
            if max_size is not None and size > max_size:
                continue
            yield (key, size)
        except Exception as e:
            if raise_error:
                raise e
            if verbose:
                logger.error(f'Error getting size of key {key}: {e}')