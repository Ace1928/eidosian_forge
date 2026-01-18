from __future__ import annotations
from collections import deque
import decimal
import gc
from itertools import chain
import random
import sys
from sys import getsizeof
import types
from . import config
from . import mock
from .. import inspect
from ..engine import Connection
from ..schema import Column
from ..schema import DropConstraint
from ..schema import DropTable
from ..schema import ForeignKeyConstraint
from ..schema import MetaData
from ..schema import Table
from ..sql import schema
from ..sql.sqltypes import Integer
from ..util import decorator
from ..util import defaultdict
from ..util import has_refcount_gc
from ..util import inspect_getfullargspec
def count_cache_key_tuples(tup):
    """given a cache key tuple, counts how many instances of actual
    tuples are found.

    used to alert large jumps in cache key complexity.

    """
    stack = [tup]
    sentinel = object()
    num_elements = 0
    while stack:
        elem = stack.pop(0)
        if elem is sentinel:
            num_elements += 1
        elif isinstance(elem, tuple):
            if elem:
                stack = list(elem) + [sentinel] + stack
    return num_elements