import datetime
from redis.utils import str_if_bytes
def parse_sentinel_masters_resp3(response):
    return [parse_sentinel_state(master) for master in response]