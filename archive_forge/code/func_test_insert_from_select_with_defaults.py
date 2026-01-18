from decimal import Decimal
import uuid
from . import testing
from .. import fixtures
from ..assertions import eq_
from ..config import requirements
from ..schema import Column
from ..schema import Table
from ... import Double
from ... import Float
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import Numeric
from ... import select
from ... import String
from ...types import LargeBinary
from ...types import UUID
from ...types import Uuid
@requirements.insert_from_select
def test_insert_from_select_with_defaults(self, connection):
    table = self.tables.includes_defaults
    connection.execute(table.insert(), [dict(id=1, data='data1'), dict(id=2, data='data2'), dict(id=3, data='data3')])
    connection.execute(table.insert().inline().from_select(('id', 'data'), select(table.c.id + 5, table.c.data).where(table.c.data.in_(['data2', 'data3']))))
    eq_(connection.execute(select(table).order_by(table.c.data, table.c.id)).fetchall(), [(1, 'data1', 5, 4), (2, 'data2', 5, 4), (7, 'data2', 5, 4), (3, 'data3', 5, 4), (8, 'data3', 5, 4)])