import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_replicate(self, target_nodes: 'TargetNodesT', node_id: str) -> ResponseT:
    """
        Reconfigure a node as a slave of the specified master node

        For more information see https://redis.io/commands/cluster-replicate
        """
    return self.execute_command('CLUSTER REPLICATE', node_id, target_nodes=target_nodes)