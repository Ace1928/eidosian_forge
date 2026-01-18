import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@contextlib.contextmanager
def shared_intermediates(cache=None):
    """Context in which intermediate results are shared.

    Note that intermediate LazyArray instances (which can reference actual
    data) will not be garbage collected until
    1. this context exits, and
    2. the yielded cache is garbage collected (if it was captured).

    Parameters
    ----------
    cache : dict
        If specified, a user-stored dict in which intermediate lazy arrays will
        be stored. This can be used to interleave sharing contexts.

    Returns
    -------
    cache : dict
        A dictionary in which sharing results are stored. If ignored,
        sharing results will be garbage collected when this context is
        exited. This dict can be passed to another context to resume
        sharing.
    """
    if cache is None:
        cache = {}
    _add_sharing_cache(cache)
    try:
        yield cache
    finally:
        _remove_sharing_cache()