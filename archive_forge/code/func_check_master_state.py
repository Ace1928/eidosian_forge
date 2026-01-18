import asyncio
import random
import weakref
from typing import AsyncIterator, Iterable, Mapping, Optional, Sequence, Tuple, Type
from redis.asyncio.client import Redis
from redis.asyncio.connection import (
from redis.commands import AsyncSentinelCommands
from redis.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from redis.utils import str_if_bytes
def check_master_state(self, state: dict, service_name: str) -> bool:
    if not state['is_master'] or state['is_sdown'] or state['is_odown']:
        return False
    if state['num-other-sentinels'] < self.min_other_sentinels:
        return False
    return True