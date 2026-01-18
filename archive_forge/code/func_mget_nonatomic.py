import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def mget_nonatomic(self, keys: KeysT, *args: KeyT) -> List[Optional[Any]]:
    """
        Splits the keys into different slots and then calls MGET
        for the keys of every slot. This operation will not be atomic
        if keys belong to more than one slot.

        Returns a list of values ordered identically to ``keys``

        For more information see https://redis.io/commands/mget
        """
    keys = list_or_args(keys, args)
    slots_to_keys = self._partition_keys_by_slot(keys)
    res = self._execute_pipeline_by_slot('MGET', slots_to_keys)
    return self._reorder_keys_by_command(keys, slots_to_keys, res)