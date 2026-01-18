import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_setslot(self, target_node: 'TargetNodesT', node_id: str, slot_id: int, state: str) -> ResponseT:
    """
        Bind an hash slot to a specific node

        :target_node: 'ClusterNode'
            The node to execute the command on

        For more information see https://redis.io/commands/cluster-setslot
        """
    if state.upper() in ('IMPORTING', 'NODE', 'MIGRATING'):
        return self.execute_command('CLUSTER SETSLOT', slot_id, state, node_id, target_nodes=target_node)
    elif state.upper() == 'STABLE':
        raise RedisError('For "stable" state please use cluster_setslot_stable')
    else:
        raise RedisError(f'Invalid slot state: {state}')