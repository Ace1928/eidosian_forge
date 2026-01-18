import contextlib
import functools
import numbers
import threading
from collections import Counter, defaultdict
from .parser import alpha_canonicalize, parse_einsum_input
@functools.wraps(to_backend)
def cached_to_backend(array):
    if not currently_sharing():
        return to_backend(array)
    key = (to_backend.__name__, id(array))
    return _memoize(key, to_backend, array)