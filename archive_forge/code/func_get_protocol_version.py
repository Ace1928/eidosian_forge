import copy
import random
import string
from typing import List, Tuple
import redis
from redis.typing import KeysT, KeyT
def get_protocol_version(client):
    if isinstance(client, redis.Redis) or isinstance(client, redis.asyncio.Redis):
        return client.connection_pool.connection_kwargs.get('protocol')
    elif isinstance(client, redis.cluster.AbstractRedisCluster):
        return client.nodes_manager.connection_kwargs.get('protocol')