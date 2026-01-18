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
def drop_all_tables_from_metadata(metadata, engine_or_connection):
    from . import engines

    def go(connection):
        engines.testing_reaper.prepare_for_drop_tables(connection)
        if not connection.dialect.supports_alter:
            from . import assertions
            with assertions.expect_warnings("Can't sort tables", assert_=False):
                metadata.drop_all(connection)
        else:
            metadata.drop_all(connection)
    if not isinstance(engine_or_connection, Connection):
        with engine_or_connection.begin() as connection:
            go(connection)
    else:
        go(engine_or_connection)