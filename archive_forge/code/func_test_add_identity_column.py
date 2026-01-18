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
def test_add_identity_column(self):
    m1 = MetaData()
    m2 = MetaData()
    Table('user', m1, Column('other', sa.Text))
    Table('user', m2, Column('other', sa.Text), Column('id', Integer, sa.Identity(start=5, increment=7), primary_key=True))
    diffs = self._fixture(m1, m2)
    eq_(diffs[0][0], 'add_column')
    eq_(diffs[0][2], 'user')
    eq_(diffs[0][3].name, 'id')
    i = diffs[0][3].identity
    is_true(isinstance(i, sa.Identity))
    eq_(i.start, 5)
    eq_(i.increment, 7)