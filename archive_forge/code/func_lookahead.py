import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
def lookahead(it):
    """Yield pairs of (current, next) items in `it`.

    `next` is None if `current` is the last item.
    Example:
        >>> list(lookahead(x for x in range(6)))
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
    """
    a, b = tee(it)
    next(b, None)
    return zip_longest(a, b)