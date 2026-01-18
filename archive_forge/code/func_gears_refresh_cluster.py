import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def gears_refresh_cluster(self, **kwargs) -> ResponseT:
    """
        On an OSS cluster, before executing any gears function, you must call this command. # noqa
        """
    return self.execute_command('REDISGEARS_2.REFRESHCLUSTER', **kwargs)