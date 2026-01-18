import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
def seq_concat_seq(a, b):
    """Concatenate two sequences: ``a + b``.

    Returns:
        Sequence: The return value will depend on the largest sequence
            - if b is larger and is a tuple, the return value will be a tuple.
            - if a is larger and is a list, the return value will be a list,
    """
    prefer = type(max([a, b], key=len))
    if not isinstance(a, prefer):
        a = prefer(a)
    if not isinstance(b, prefer):
        b = prefer(b)
    return a + b