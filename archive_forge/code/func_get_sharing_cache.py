import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def get_sharing_cache():
    """Return the most recent sharing cache -- thread specific."""
    return _SHARING_STACK[threading.get_ident()][-1]