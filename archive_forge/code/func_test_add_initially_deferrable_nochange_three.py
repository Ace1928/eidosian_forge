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
@config.requirements.fk_initially
@config.requirements.fk_deferrable
def test_add_initially_deferrable_nochange_three(self):
    diffs = self._fk_opts_fixture({'deferrable': None, 'initially': 'deferred'}, {'deferrable': None, 'initially': 'deferred'})
    eq_(diffs, [])