import asyncio
import collections
import random
import socket
import ssl
import warnings
from typing import (
from redis._parsers import AsyncCommandsParser, Encoder
from redis._parsers.helpers import (
from redis.asyncio.client import ResponseCallbackT
from redis.asyncio.connection import Connection, DefaultParser, SSLConnection, parse_url
from redis.asyncio.lock import Lock
from redis.asyncio.retry import Retry
from redis.backoff import default_backoff
from redis.client import EMPTY_RESPONSE, NEVER_DECODE, AbstractRedis
from redis.cluster import (
from redis.commands import READ_COMMANDS, AsyncRedisClusterCommands
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.typing import AnyKeyT, EncodableT, KeyT
from redis.utils import (
def set_nodes(self, old: Dict[str, 'ClusterNode'], new: Dict[str, 'ClusterNode'], remove_old: bool=False) -> None:
    if remove_old:
        for name in list(old.keys()):
            if name not in new:
                task = asyncio.create_task(old.pop(name).disconnect())
    for name, node in new.items():
        if name in old:
            if old[name] is node:
                continue
            task = asyncio.create_task(old[name].disconnect())
        old[name] = node