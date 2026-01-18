import os
import sys
import time
from alembic import command as alembic_command
from oslo_config import cfg
from oslo_db import exception as db_exc
from oslo_log import log as logging
from oslo_utils import encodeutils
from glance.common import config
from glance.common import exception
from glance import context
from glance.db import migration as db_migration
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import metadata
from glance.i18n import _
@args('--age_in_days', type=int, help='Purge deleted rows older than age in days')
@args('--max_rows', type=int, help='Limit number of records to delete. All deleted rows will be purged if equals -1.')
def purge_images_table(self, age_in_days=180, max_rows=100):
    """Purge deleted rows older than a given age from images table."""
    self._purge(age_in_days, max_rows, purge_images_only=True)