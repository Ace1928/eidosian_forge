import datetime
from redis.utils import str_if_bytes
def parse_hscan(response, **options):
    cursor, r = response
    return (int(cursor), r and pairs_to_dict(r) or {})