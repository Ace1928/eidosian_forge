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
@testing.combinations(lambda: (None, sa.Computed('bar*5')), lambda: (sa.Computed('bar*5'), None), lambda: (sa.Computed('bar*5'), sa.Computed('bar * 42', persisted=True)), lambda: (sa.Computed('bar*5'), sa.Computed('bar * 42')))
@config.requirements.computed_reflects_normally
def test_cant_change_computed_warning(self, test_case):
    arg_before, arg_after = testing.resolve_lambda(test_case, **locals())
    m1 = MetaData()
    m2 = MetaData()
    arg_before = [] if arg_before is None else [arg_before]
    arg_after = [] if arg_after is None else [arg_after]
    Table('user', m1, Column('id', Integer, primary_key=True), Column('bar', Integer), Column('foo', Integer, *arg_before))
    Table('user', m2, Column('id', Integer, primary_key=True), Column('bar', Integer), Column('foo', Integer, *arg_after))
    with mock.patch('alembic.util.warn') as mock_warn:
        diffs = self._fixture(m1, m2)
    eq_(mock_warn.mock_calls, [mock.call('Computed default on user.foo cannot be modified')])
    eq_(list(diffs), [])