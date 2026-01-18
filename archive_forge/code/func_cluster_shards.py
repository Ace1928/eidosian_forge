import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_shards(self, target_nodes=None):
    """
        Returns details about the shards of the cluster.

        For more information see https://redis.io/commands/cluster-shards
        """
    return self.execute_command('CLUSTER SHARDS', target_nodes=target_nodes)