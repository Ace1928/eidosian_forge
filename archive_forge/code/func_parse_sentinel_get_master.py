import datetime
from redis.utils import str_if_bytes
def parse_sentinel_get_master(response):
    return response and (response[0], int(response[1])) or None