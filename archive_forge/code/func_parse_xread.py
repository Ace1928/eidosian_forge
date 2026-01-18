import datetime
from redis.utils import str_if_bytes
def parse_xread(response):
    if response is None:
        return []
    return [[r[0], parse_stream_list(r[1])] for r in response]