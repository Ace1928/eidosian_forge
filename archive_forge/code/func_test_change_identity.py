import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table
from alembic.util import sqla_compat
from ._autogen_fixtures import AutogenFixtureTest
from ... import testing
from ...testing import config
from ...testing import eq_
from ...testing import is_true
from ...testing import TestBase
@testing.combinations((None, dict(start=2)), (dict(start=2), None), (dict(start=2), dict(start=2, increment=7)), (dict(always=False), dict(always=True)), (dict(start=1, minvalue=0, maxvalue=100, cycle=True), dict(start=1, minvalue=0, maxvalue=100, cycle=False)), (dict(start=10, increment=3, maxvalue=9999), dict(start=10, increment=1, maxvalue=3333)))
@config.requirements.identity_columns_alter
def test_change_identity(self, before, after):
    arg_before = (sa.Identity(**before),) if before else ()
    arg_after = (sa.Identity(**after),) if after else ()
    m1 = MetaData()
    m2 = MetaData()
    Table('user', m1, Column('id', Integer, *arg_before), Column('other', sa.Text))
    Table('user', m2, Column('id', Integer, *arg_after), Column('other', sa.Text))
    diffs = self._fixture(m1, m2)
    eq_(len(diffs[0]), 1)
    diffs = diffs[0][0]
    eq_(diffs[0], 'modify_default')
    eq_(diffs[2], 'user')
    eq_(diffs[3], 'id')
    old = diffs[5]
    new = diffs[6]

    def check(kw, idt):
        if kw:
            is_true(isinstance(idt, sa.Identity))
            for k, v in kw.items():
                eq_(getattr(idt, k), v)
        else:
            is_true(idt in (None, False))
    check(before, old)
    check(after, new)