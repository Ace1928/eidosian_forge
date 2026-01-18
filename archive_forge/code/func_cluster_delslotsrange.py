import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_delslotsrange(self, *slots: EncodableT) -> ResponseT:
    """
        Similar to the CLUSTER DELSLOTS command.
        The difference is that CLUSTER DELSLOTS takes a list of hash slots to remove
        from the node, while CLUSTER DELSLOTSRANGE takes a list of slot ranges to remove
        from the node.

        For more information see https://redis.io/commands/cluster-delslotsrange
        """
    return self.execute_command('CLUSTER DELSLOTSRANGE', *slots)