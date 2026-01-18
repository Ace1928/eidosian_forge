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
def assertTableDoesNotExist(self, table_name):
    """Assert that a given table exists cannot be selected by name."""
    try:
        sqlalchemy.Table(table_name, self.metadata, autoload_with=self.engine)
    except sqlalchemy.exc.NoSuchTableError:
        pass
    else:
        raise AssertionError('Table "%s" already exists' % table_name)