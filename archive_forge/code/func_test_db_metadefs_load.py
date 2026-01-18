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
def test_db_metadefs_load(self):
    db_metadata.db_load_metadefs = mock.Mock()
    self._main_test_helper(['glance.cmd.manage', 'db', 'load_metadefs'], db_metadata.db_load_metadefs, db_api.get_engine(), None, False, False, False)