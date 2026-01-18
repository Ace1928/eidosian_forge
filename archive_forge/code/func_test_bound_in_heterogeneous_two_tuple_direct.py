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
@testing.requires.tuple_in
def test_bound_in_heterogeneous_two_tuple_direct(self):
    table = self.tables.some_table
    stmt = select(table.c.id).where(tuple_(table.c.x, table.c.z).in_([(2, 'z2'), (3, 'z3'), (4, 'z4')])).order_by(table.c.id)
    self._assert_result(stmt, [(2,), (3,), (4,)])