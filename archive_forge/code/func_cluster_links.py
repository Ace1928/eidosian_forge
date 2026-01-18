import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_links(self, target_node: 'TargetNodesT') -> ResponseT:
    """
        Each node in a Redis Cluster maintains a pair of long-lived TCP link with each
        peer in the cluster: One for sending outbound messages towards the peer and one
        for receiving inbound messages from the peer.

        This command outputs information of all such peer links as an array.

        For more information see https://redis.io/commands/cluster-links
        """
    return self.execute_command('CLUSTER LINKS', target_nodes=target_node)