from .. import fixtures
from ..assertions import eq_
from ..schema import Column
from ..schema import Table
from ... import Integer
from ... import String
from ... import testing
@testing.variation('criteria', ['rows', 'norows', 'emptyin'])
@testing.requires.delete_returning
def test_delete_returning(self, connection, criteria):
    t = self.tables.plain_pk
    stmt = t.delete().returning(t.c.id, t.c.data)
    if criteria.norows:
        stmt = stmt.where(t.c.id == 10)
    elif criteria.rows:
        stmt = stmt.where(t.c.id == 2)
    elif criteria.emptyin:
        stmt = stmt.where(t.c.id.in_([]))
    else:
        criteria.fail()
    r = connection.execute(stmt)
    assert not r.is_insert
    assert r.returns_rows
    eq_(r.keys(), ['id', 'data'])
    if criteria.rows:
        eq_(r.all(), [(2, 'd2')])
    else:
        eq_(r.all(), [])
    eq_(connection.execute(t.select().order_by(t.c.id)).fetchall(), [(1, 'd1'), (3, 'd3')] if criteria.rows else [(1, 'd1'), (2, 'd2'), (3, 'd3')])