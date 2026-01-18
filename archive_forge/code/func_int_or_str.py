from aioredis.client import Redis, StrictRedis
from aioredis.connection import (
from aioredis.exceptions import (
from aioredis.utils import from_url
def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value