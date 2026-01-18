import os
import sys
from oslo_config import cfg
from oslo_db import options as db_options
from glance.common import utils
from glance.db import migration as db_migration
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.db.sqlalchemy import api as db_api
from glance.tests import functional
from glance.tests.utils import depends_on_exe
from glance.tests.utils import execute
from glance.tests.utils import skip_if_disabled
@depends_on_exe('sqlite3')
@skip_if_disabled
def test_db_creation(self):
    """Test schema creation by db_sync on a fresh DB"""
    self._db_command(db_method='sync')
    for table in ['images', 'image_tags', 'image_locations', 'image_members', 'image_properties']:
        self._assert_table_exists(table)