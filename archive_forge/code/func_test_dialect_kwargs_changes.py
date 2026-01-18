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
def test_dialect_kwargs_changes(self):
    m1 = MetaData()
    m2 = MetaData()
    if sqla_compat.identity_has_dialect_kwargs:
        args = {'oracle_on_null': True, 'oracle_order': True}
    else:
        args = {'on_null': True, 'order': True}
    Table('user', m1, Column('id', Integer, sa.Identity(start=2)))
    id_ = sa.Identity(start=2, **args)
    Table('user', m2, Column('id', Integer, id_))
    diffs = self._fixture(m1, m2)
    if config.db.name == 'oracle':
        is_true(len(diffs), 1)
        eq_(diffs[0][0][0], 'modify_default')
    else:
        eq_(diffs, [])