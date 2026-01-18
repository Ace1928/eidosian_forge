import datetime
from redis.utils import str_if_bytes
def parse_zadd(response, **options):
    if response is None:
        return None
    if options.get('as_score'):
        return float(response)
    return int(response)