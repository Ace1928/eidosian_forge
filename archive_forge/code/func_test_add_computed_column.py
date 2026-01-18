import sqlalchemy as sa
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table
from ._autogen_fixtures import AutogenFixtureTest
from ... import testing
from ...testing import config
from ...testing import eq_
from ...testing import exclusions
from ...testing import is_
from ...testing import is_true
from ...testing import mock
from ...testing import TestBase
def test_add_computed_column(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('user', m1, Column('id', Integer, primary_key=True))
    Table('user', m2, Column('id', Integer, primary_key=True), Column('foo', Integer, sa.Computed('5')))
    diffs = self._fixture(m1, m2)
    eq_(diffs[0][0], 'add_column')
    eq_(diffs[0][2], 'user')
    eq_(diffs[0][3].name, 'foo')
    c = diffs[0][3].computed
    is_true(isinstance(c, sa.Computed))
    is_(c.persisted, None)
    eq_(str(c.sqltext), '5')