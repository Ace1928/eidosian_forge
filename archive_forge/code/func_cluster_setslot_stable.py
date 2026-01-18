import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_setslot_stable(self, slot_id: int) -> ResponseT:
    """
        Clears migrating / importing state from the slot.
        It determines by it self what node the slot is in and sends it there.

        For more information see https://redis.io/commands/cluster-setslot
        """
    return self.execute_command('CLUSTER SETSLOT', slot_id, 'STABLE')