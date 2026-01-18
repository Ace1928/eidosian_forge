import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_failover(self, target_node: 'TargetNodesT', option: Optional[str]=None) -> ResponseT:
    """
        Forces a slave to perform a manual failover of its master
        Sends to specified node

        :target_node: 'ClusterNode'
            The node to execute the command on

        For more information see https://redis.io/commands/cluster-failover
        """
    if option:
        if option.upper() not in ['FORCE', 'TAKEOVER']:
            raise RedisError(f'Invalid option for CLUSTER FAILOVER command: {option}')
        else:
            return self.execute_command('CLUSTER FAILOVER', option, target_nodes=target_node)
    else:
        return self.execute_command('CLUSTER FAILOVER', target_nodes=target_node)