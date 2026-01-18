from sqlalchemy import BigInteger
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy.testing import in_
from ._autogen_fixtures import AutogenFixtureTest
from ... import testing
from ...testing import config
from ...testing import eq_
from ...testing import is_
from ...testing import TestBase
@testing.combinations((True,), (False,))
@config.requirements.comments
def test_all_existings_filled(self, pk):
    m1 = MetaData()
    m2 = MetaData()
    Table('a', m1, Column('x', Integer, primary_key=pk))
    Table('a', m2, Column('x', Integer, comment='x', primary_key=pk))
    alter_col = self._assert_alter_col(m1, m2, pk)
    eq_(alter_col.modify_comment, 'x')