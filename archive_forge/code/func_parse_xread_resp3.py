import datetime
from redis.utils import str_if_bytes
def parse_xread_resp3(response):
    if response is None:
        return {}
    return {key: [parse_stream_list(value)] for key, value in response.items()}