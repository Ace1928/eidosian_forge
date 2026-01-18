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
def test_composed_multiple(self):
    table = self.tables.some_table
    lx = (table.c.x + table.c.y).label('lx')
    ly = (func.lower(table.c.q) + table.c.p).label('ly')
    self._assert_result(select(lx, ly).order_by(lx, ly.desc()), [(3, 'q1p3'), (5, 'q2p2'), (7, 'q3p1')])