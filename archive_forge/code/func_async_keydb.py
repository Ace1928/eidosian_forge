import typing
import logging
import threading
from aiokeydb.v1.lock import Lock
from aiokeydb.v1.connection import Encoder, ConnectionPool
from aiokeydb.v1.core import KeyDB, PubSub, Pipeline
from aiokeydb.v1.typing import Number, KeyT, AbsExpiryT, ExpiryT
from aiokeydb.v1.asyncio.lock import AsyncLock
from aiokeydb.v1.asyncio.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline
from aiokeydb.v1.asyncio.connection import AsyncConnectionPool
from aiokeydb.v1.client.config import KeyDBSettings
from aiokeydb.v1.client.types import classproperty, KeyDBUri
from aiokeydb.v1.client.schemas.session import KeyDBSession
from aiokeydb.v1.client.serializers import SerializerType
@classproperty
def async_keydb(cls) -> AsyncKeyDB:
    """
        returns the underlying `AsyncKeyDB` client from within
        the current session.
        """
    return cls.get_session().async_client