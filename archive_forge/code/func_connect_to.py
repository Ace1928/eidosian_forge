import random
import weakref
from typing import Optional
from redis.client import Redis
from redis.commands import SentinelCommands
from redis.connection import Connection, ConnectionPool, SSLConnection
from redis.exceptions import ConnectionError, ReadOnlyError, ResponseError, TimeoutError
from redis.utils import str_if_bytes
def connect_to(self, address):
    self.host, self.port = address
    super().connect()
    if self.connection_pool.check_connection:
        self.send_command('PING')
        if str_if_bytes(self.read_response()) != 'PONG':
            raise ConnectionError('PING failed')