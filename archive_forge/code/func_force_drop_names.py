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
def force_drop_names(*names):
    """Force the given table names to be dropped after test complete,
    isolating for foreign key cycles

    """

    @decorator
    def go(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        finally:
            drop_all_tables(config.db, inspect(config.db), include_names=names)
    return go