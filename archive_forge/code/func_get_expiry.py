import enum
import functools
import redis
from redis import exceptions as redis_exceptions
def get_expiry(client, key, prior_version=None):
    """Gets an expiry for a key (using **best** determined ttl method)."""
    is_new_enough, _prior_version = is_server_new_enough(client, (2, 6), prior_version=prior_version)
    if is_new_enough:
        result = client.pttl(key)
        try:
            return _UNKNOWN_EXPIRE_MAPPING[result]
        except KeyError:
            return result / 1000.0
    else:
        result = client.ttl(key)
        try:
            return _UNKNOWN_EXPIRE_MAPPING[result]
        except KeyError:
            return float(result)