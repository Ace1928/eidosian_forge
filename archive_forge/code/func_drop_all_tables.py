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
def drop_all_tables(engine, inspector, schema=None, consider_schemas=(None,), include_names=None):
    if include_names is not None:
        include_names = set(include_names)
    if schema is not None:
        assert consider_schemas == (None,), 'consider_schemas and schema are mutually exclusive'
        consider_schemas = (schema,)
    with engine.begin() as conn:
        for table_key, fkcs in reversed(inspector.sort_tables_on_foreign_key_dependency(consider_schemas=consider_schemas)):
            if table_key:
                if include_names is not None and table_key[1] not in include_names:
                    continue
                conn.execute(DropTable(Table(table_key[1], MetaData(), schema=table_key[0])))
            elif fkcs:
                if not engine.dialect.supports_alter:
                    continue
                for t_key, fkc in fkcs:
                    if include_names is not None and t_key[1] not in include_names:
                        continue
                    tb = Table(t_key[1], MetaData(), Column('x', Integer), Column('y', Integer), schema=t_key[0])
                    conn.execute(DropConstraint(ForeignKeyConstraint([tb.c.x], [tb.c.y], name=fkc)))