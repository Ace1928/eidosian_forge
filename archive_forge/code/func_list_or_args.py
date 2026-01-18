import copy
import random
import string
from typing import List, Tuple
import redis
from redis.typing import KeysT, KeyT
def list_or_args(keys: KeysT, args: Tuple[KeyT, ...]) -> List[KeyT]:
    try:
        iter(keys)
        if isinstance(keys, (bytes, str)):
            keys = [keys]
        else:
            keys = list(keys)
    except TypeError:
        keys = [keys]
    if args:
        keys.extend(args)
    return keys