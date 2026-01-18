import enum
import functools
import redis
from redis import exceptions as redis_exceptions
def apply_expiry(client, key, expiry, prior_version=None):
    """Applies an expiry to a key (using **best** determined expiry method)."""
    is_new_enough, _prior_version = is_server_new_enough(client, (2, 6), prior_version=prior_version)
    if is_new_enough:
        ms_expiry = expiry * 1000.0
        ms_expiry = max(0, int(ms_expiry))
        result = client.pexpire(key, ms_expiry)
    else:
        sec_expiry = int(expiry)
        sec_expiry = max(0, sec_expiry)
        result = client.expire(key, sec_expiry)
    return bool(result)