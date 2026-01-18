import datetime
from redis.utils import str_if_bytes
def bool_ok(response, **options):
    return str_if_bytes(response) == 'OK'