import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def cluster_count_failure_report(self, node_id: str) -> ResponseT:
    """
        Return the number of failure reports active for a given node
        Sends to a random node

        For more information see https://redis.io/commands/cluster-count-failure-reports
        """
    return self.execute_command('CLUSTER COUNT-FAILURE-REPORTS', node_id)