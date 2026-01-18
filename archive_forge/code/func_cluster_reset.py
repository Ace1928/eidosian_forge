import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_reset(self, soft: bool=True, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
    """
        Reset a Redis Cluster node

        If 'soft' is True then it will send 'SOFT' argument
        If 'soft' is False then it will send 'HARD' argument

        For more information see https://redis.io/commands/cluster-reset
        """
    return self.execute_command('CLUSTER RESET', b'SOFT' if soft else b'HARD', target_nodes=target_nodes)