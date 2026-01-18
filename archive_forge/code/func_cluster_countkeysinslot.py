import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_countkeysinslot(self, slot_id: int) -> ResponseT:
    """
        Return the number of local keys in the specified hash slot
        Send to node based on specified slot_id

        For more information see https://redis.io/commands/cluster-countkeysinslot
        """
    return self.execute_command('CLUSTER COUNTKEYSINSLOT', slot_id)