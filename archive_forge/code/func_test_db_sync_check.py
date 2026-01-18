import fixtures
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures as db_fixtures
from oslo_log import fixture as log_fixture
from oslo_log import log
import sqlalchemy.exc
from keystone.cmd import cli
from keystone.common import sql
from keystone.common.sql import upgrades
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
def test_db_sync_check(self):
    checker = cli.DbSync()
    log_info = self.useFixture(fixtures.FakeLogger(level=log.INFO))
    status = checker.check_db_sync_status()
    self.assertIn('keystone-manage db_sync --expand', log_info.output)
    self.assertEqual(status, 2)
    self.expand()
    log_info = self.useFixture(fixtures.FakeLogger(level=log.INFO))
    status = checker.check_db_sync_status()
    self.assertIn('keystone-manage db_sync --contract', log_info.output)
    self.assertEqual(status, 4)
    self.contract()
    log_info = self.useFixture(fixtures.FakeLogger(level=log.INFO))
    status = checker.check_db_sync_status()
    self.assertIn('All db_sync commands are upgraded', log_info.output)
    self.assertEqual(status, 0)