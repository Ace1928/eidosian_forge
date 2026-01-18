import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_myid(self, target_node: 'TargetNodesT') -> ResponseT:
    """
        Returns the node's id.

        :target_node: 'ClusterNode'
            The node to execute the command on

        For more information check https://redis.io/commands/cluster-myid/
        """
    return self.execute_command('CLUSTER MYID', target_nodes=target_node)