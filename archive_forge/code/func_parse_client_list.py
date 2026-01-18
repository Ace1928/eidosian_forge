import datetime
from redis.utils import str_if_bytes
def parse_client_list(response, **options):
    clients = []
    for c in str_if_bytes(response).splitlines():
        clients.append(dict((pair.split('=', 1) for pair in c.split(' '))))
    return clients