import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def maybe_materialize(x):
    """Recursively evaluate LazyArray instances in tuples, lists and dicts."""
    try:
        return _materialize_dispatch[x.__class__](x)
    except KeyError:
        _materialize_dispatch[x.__class__] = materialize_identity
        return x