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
def mset_nonatomic(self, mapping: Mapping[AnyKeyT, EncodableT]) -> 'ClusterPipeline':
    encoder = self._client.encoder
    slots_pairs = {}
    for pair in mapping.items():
        slot = key_slot(encoder.encode(pair[0]))
        slots_pairs.setdefault(slot, []).extend(pair)
    for pairs in slots_pairs.values():
        self.execute_command('MSET', *pairs)
    return self