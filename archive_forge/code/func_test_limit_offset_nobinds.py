import collections.abc as collections_abc
import itertools
from .. import AssertsCompiledSQL
from .. import AssertsExecutionResults
from .. import config
from .. import fixtures
from ..assertions import assert_raises
from ..assertions import eq_
from ..assertions import in_
from ..assertsql import CursorSQL
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import case
from ... import column
from ... import Computed
from ... import exists
from ... import false
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import null
from ... import select
from ... import String
from ... import table
from ... import testing
from ... import text
from ... import true
from ... import tuple_
from ... import TupleType
from ... import union
from ... import values
from ...exc import DatabaseError
from ...exc import ProgrammingError
@testing.requires.offset
def test_limit_offset_nobinds(self):
    """test that 'literal binds' mode works - no bound params."""
    table = self.tables.some_table
    stmt = select(table).order_by(table.c.id).limit(2).offset(1)
    sql = stmt.compile(dialect=config.db.dialect, compile_kwargs={'literal_binds': True})
    sql = str(sql)
    self._assert_result_str(sql, [(2, 2, 3), (3, 3, 4)])