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
def test_bound_in_heterogeneous_two_tuple_typed_bindparam_non_tuple(self):

    class LikeATuple(collections_abc.Sequence):

        def __init__(self, *data):
            self._data = data

        def __iter__(self):
            return iter(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

        def __len__(self):
            return len(self._data)
    stmt = text('select id FROM some_table WHERE (x, z) IN :q ORDER BY id').bindparams(bindparam('q', type_=TupleType(Integer(), String()), expanding=True))
    self._assert_result(stmt, [(2,), (3,), (4,)], params={'q': [LikeATuple(2, 'z2'), LikeATuple(3, 'z3'), LikeATuple(4, 'z4')]})