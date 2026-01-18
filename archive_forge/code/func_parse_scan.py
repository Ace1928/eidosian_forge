import datetime
from redis.utils import str_if_bytes
def parse_scan(response, **options):
    cursor, r = response
    return (int(cursor), r)