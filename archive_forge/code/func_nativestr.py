import copy
import random
import string
from typing import List, Tuple
import redis
from redis.typing import KeysT, KeyT
def nativestr(x):
    """Return the decoded binary string, or a string, depending on type."""
    r = x.decode('utf-8', 'replace') if isinstance(x, bytes) else x
    if r == 'null':
        return
    return r