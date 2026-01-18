import io
from unittest import mock
import fixtures
from glance.cmd import manage
from glance.common import exception
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import metadata as db_metadata
from glance.tests import utils as test_utils
from sqlalchemy.engine.url import make_url as sqlalchemy_make_url
@mock.patch.object(manage.DbCommands, 'sync')
def test_legacy_db_sync_version(self, db_sync):
    self._main_test_helper(['glance.cmd.manage', 'db_sync', 'liberty'], manage.DbCommands.sync, 'liberty')