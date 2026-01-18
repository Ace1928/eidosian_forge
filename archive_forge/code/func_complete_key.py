import os
import gc
import json
import time
import asyncio
import typing
import anyio
import threading
from contextlib import asynccontextmanager, suppress
from lazyops.utils.logs import default_logger as logger
from lazyops.utils.serialization import ObjectEncoder
from lazyops.utils.helpers import create_background_task
from aiokeydb.v2.types.base import BaseModel, Field, validator
from aiokeydb.v2.types.base import KeyDBUri, lazyproperty
from aiokeydb.v2.types.session import KeyDBSession
from aiokeydb.v2.serializers import SerializerType
from aiokeydb.v2.client import KeyDBClient
from aiokeydb.v2.commands import AsyncScript
from aiokeydb.v2.connection import (
from aiokeydb.v2.types.jobs import (
import aiokeydb.v2.exceptions as exceptions
from aiokeydb.v2.exceptions import JobError
from aiokeydb.v2.utils.queue import (
from aiokeydb.v2.configs import settings
from aiokeydb.v2.utils import set_ulimits, get_ulimits
from aiokeydb.v2.backoff import default_backoff
from redis.asyncio.retry import Retry
from typing import TYPE_CHECKING, overload
@lazyproperty
def complete_key(self) -> str:
    return f'{self.prefix}:{self.name}:complete'