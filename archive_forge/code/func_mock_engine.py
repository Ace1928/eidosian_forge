from __future__ import annotations
import collections
import re
import typing
from typing import Any
from typing import Dict
from typing import Optional
import warnings
import weakref
from . import config
from .util import decorator
from .util import gc_collect
from .. import event
from .. import pool
from ..util import await_only
from ..util.typing import Literal
def mock_engine(dialect_name=None):
    """Provides a mocking engine based on the current testing.db.

    This is normally used to test DDL generation flow as emitted
    by an Engine.

    It should not be used in other cases, as assert_compile() and
    assert_sql_execution() are much better choices with fewer
    moving parts.

    """
    from sqlalchemy import create_mock_engine
    if not dialect_name:
        dialect_name = config.db.name
    buffer = []

    def executor(sql, *a, **kw):
        buffer.append(sql)

    def assert_sql(stmts):
        recv = [re.sub('[\\n\\t]', '', str(s)) for s in buffer]
        assert recv == stmts, recv

    def print_sql():
        d = engine.dialect
        return '\n'.join((str(s.compile(dialect=d)) for s in engine.mock))
    engine = create_mock_engine(dialect_name + '://', executor)
    assert not hasattr(engine, 'mock')
    engine.mock = buffer
    engine.assert_sql = assert_sql
    engine.print_sql = print_sql
    return engine