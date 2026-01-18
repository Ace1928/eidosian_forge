import random
import weakref
from typing import Optional
from redis.client import Redis
from redis.commands import SentinelCommands
from redis.connection import Connection, ConnectionPool, SSLConnection
from redis.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from redis.utils import str_if_bytes
def discover_slaves(self, service_name):
    """Returns a list of alive slaves for service ``service_name``"""
    for sentinel in self.sentinels:
        try:
            slaves = sentinel.sentinel_slaves(service_name)
        except (ConnectionError, ResponseError, TimeoutError):
            continue
        slaves = self.filter_slaves(slaves)
        if slaves:
            return slaves
    return []