import copy
import random
import string
from typing import List, Tuple
import redis
from redis.typing import KeysT, KeyT
def delist(x):
    """Given a list of binaries, return the stringified version."""
    if x is None:
        return x
    return [nativestr(obj) for obj in x]