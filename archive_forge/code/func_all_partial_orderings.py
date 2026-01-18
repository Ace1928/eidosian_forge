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
def all_partial_orderings(tuples, elements):
    edges = defaultdict(set)
    for parent, child in tuples:
        edges[child].add(parent)

    def _all_orderings(elements):
        if len(elements) == 1:
            yield list(elements)
        else:
            for elem in elements:
                subset = set(elements).difference([elem])
                if not subset.intersection(edges[elem]):
                    for sub_ordering in _all_orderings(subset):
                        yield ([elem] + sub_ordering)
    return iter(_all_orderings(elements))