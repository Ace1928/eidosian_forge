import asyncio
import random
import weakref
from typing import AsyncIterator, Iterable, Mapping, Optional, Sequence, Tuple, Type
from redis.asyncio.client import Redis
from redis.asyncio.connection import (
from redis.commands import AsyncSentinelCommands
from redis.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from redis.utils import str_if_bytes
def filter_slaves(self, slaves: Iterable[Mapping]) -> Sequence[Tuple[EncodableT, EncodableT]]:
    """Remove slaves that are in an ODOWN or SDOWN state"""
    slaves_alive = []
    for slave in slaves:
        if slave['is_odown'] or slave['is_sdown']:
            continue
        slaves_alive.append((slave['ip'], slave['port']))
    return slaves_alive