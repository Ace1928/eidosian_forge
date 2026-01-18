import datetime
from redis.utils import str_if_bytes
def parse_xclaim(response, **options):
    if options.get('parse_justid', False):
        return response
    return parse_stream_list(response)