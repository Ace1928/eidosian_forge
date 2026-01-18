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
@decorator
def provide_metadata(fn, *args, **kw):
    """Provide bound MetaData for a single test, dropping afterwards.

    Legacy; use the "metadata" pytest fixture.

    """
    from . import fixtures
    metadata = schema.MetaData()
    self = args[0]
    prev_meta = getattr(self, 'metadata', None)
    self.metadata = metadata
    try:
        return fn(*args, **kw)
    finally:
        fixtures.close_all_sessions()
        cfc = fixtures.base._connection_fixture_connection
        if cfc:
            drop_all_tables_from_metadata(metadata, cfc)
            cfc.get_transaction().commit()
        else:
            drop_all_tables_from_metadata(metadata, config.db)
        self.metadata = prev_meta