import datetime
from redis.utils import str_if_bytes
def parse_debug_object(response):
    """Parse the results of Redis's DEBUG OBJECT command into a Python dict"""
    response = str_if_bytes(response)
    response = 'type:' + response
    response = dict((kv.split(':') for kv in response.split()))
    int_fields = ('refcount', 'serializedlength', 'lru', 'lru_seconds_idle')
    for field in int_fields:
        if field in response:
            response[field] = int(response[field])
    return response