from .. import fixtures
from ..assertions import eq_
from ..schema import Column
from ..schema import Table
from ... import ForeignKey
from ... import Integer
from ... import select
from ... import String
from ... import testing
def test_insert_from_select_round_trip(self, connection):
    some_table = self.tables.some_table
    some_other_table = self.tables.some_other_table
    cte = select(some_table).where(some_table.c.data.in_(['d2', 'd3', 'd4'])).cte('some_cte')
    connection.execute(some_other_table.insert().from_select(['id', 'data', 'parent_id'], select(cte)))
    eq_(connection.execute(select(some_other_table).order_by(some_other_table.c.id)).fetchall(), [(2, 'd2', 1), (3, 'd3', 1), (4, 'd4', 3)])