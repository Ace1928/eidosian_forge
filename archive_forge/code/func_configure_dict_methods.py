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
from aiokeydb.typing import Number, KeyT, ExpiryT, AbsExpiryT, PatternT
from aiokeydb.lock import Lock, AsyncLock
from aiokeydb.core import KeyDB, PubSub, Pipeline, PipelineT, PubSubT
from aiokeydb.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline, AsyncPipelineT, AsyncPubSubT
from aiokeydb.connection import Encoder, ConnectionPool, AsyncConnectionPool
from aiokeydb.exceptions import (
from aiokeydb.types import KeyDBUri, ENOVAL
from aiokeydb.configs import KeyDBSettings
from aiokeydb.utils import full_name, args_to_key, get_keydb_settings
from aiokeydb.utils.helpers import create_retryable_client, afail_after
from aiokeydb.utils.logs import logger
from .cachify import cachify, create_cachify, FT
from aiokeydb.serializers import BaseSerializer
from inspect import iscoroutinefunction
def configure_dict_methods(self, method: typing.Optional[str]=None, async_enabled: typing.Optional[bool]=None):
    """
        Configures the Dict get/set methods
        """
    _configure_dict_methods(self, method=method, async_enabled=async_enabled)