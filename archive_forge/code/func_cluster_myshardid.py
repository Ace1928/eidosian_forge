import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_myshardid(self, target_nodes=None):
    """
        Returns the shard ID of the node.

        For more information see https://redis.io/commands/cluster-myshardid/
        """
    return self.execute_command('CLUSTER MYSHARDID', target_nodes=target_nodes)