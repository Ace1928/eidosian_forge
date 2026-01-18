import datetime
from redis.utils import str_if_bytes
def parse_xautoclaim(response, **options):
    if options.get('parse_justid', False):
        return response[1]
    response[1] = parse_stream_list(response[1])
    return response