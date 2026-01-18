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
def test_all_existings_filled_in_server_default(self, pk):
    m1 = MetaData()
    m2 = MetaData()
    Table('a', m1, Column('x', Integer, server_default='5', primary_key=pk))
    Table('a', m2, Column('x', Integer, server_default='5', comment='new', primary_key=pk))
    alter_col = self._assert_alter_col(m1, m2, pk)
    in_('5', alter_col.existing_server_default.arg.text)