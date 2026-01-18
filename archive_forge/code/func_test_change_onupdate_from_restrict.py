from sqlalchemy import Column
from sqlalchemy import ForeignKeyConstraint
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from ._autogen_fixtures import AutogenFixtureTest
from ...testing import combinations
from ...testing import config
from ...testing import eq_
from ...testing import mock
from ...testing import TestBase
@config.requirements.fk_ondelete_restrict
def test_change_onupdate_from_restrict(self):
    """test the RESTRICT option which MySQL doesn't report on"""
    diffs = self._fk_opts_fixture({'onupdate': 'restrict'}, {'onupdate': 'cascade'})
    self._assert_fk_diff(diffs[0], 'remove_fk', 'user', ['tid'], 'some_table', ['id'], onupdate=mock.ANY, ondelete=None, conditional_name='servergenerated')
    self._assert_fk_diff(diffs[1], 'add_fk', 'user', ['tid'], 'some_table', ['id'], onupdate='cascade', ondelete=None)