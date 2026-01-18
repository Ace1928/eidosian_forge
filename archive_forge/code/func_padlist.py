import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
def padlist(container, size, default=None):
    """Pad list with default elements.

    Example:
        >>> first, last, city = padlist(['George', 'Costanza', 'NYC'], 3)
        ('George', 'Costanza', 'NYC')
        >>> first, last, city = padlist(['George', 'Costanza'], 3)
        ('George', 'Costanza', None)
        >>> first, last, city, planet = padlist(
        ...     ['George', 'Costanza', 'NYC'], 4, default='Earth',
        ... )
        ('George', 'Costanza', 'NYC', 'Earth')
    """
    return list(container)[:size] + [default] * (size - len(container))