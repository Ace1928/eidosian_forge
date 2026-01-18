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
@config.requirements.fk_onupdate
def test_nochange_onupdate_noaction(self):
    """test the NO ACTION option which generally comes back as None"""
    diffs = self._fk_opts_fixture({'onupdate': 'no action'}, {'onupdate': 'no action'})
    eq_(diffs, [])