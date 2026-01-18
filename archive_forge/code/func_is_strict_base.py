from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
def is_strict_base(typ):
    for other in types:
        if typ != other and typ in other.__mro__:
            return True
    return False