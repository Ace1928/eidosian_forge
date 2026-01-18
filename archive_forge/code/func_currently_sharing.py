import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def currently_sharing():
    """Check if we are currently sharing a cache -- thread specific."""
    return threading.get_ident() in _SHARING_STACK