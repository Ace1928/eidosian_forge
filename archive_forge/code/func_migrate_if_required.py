import datetime
import os
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance import context
import glance.db
from glance.i18n import _
from glance.image_cache.drivers import common
def migrate_if_required():
    if can_migrate_to_central_db():
        sqlite_db_file = get_db_path()
        if sqlite_db_file:
            LOG.info('Initiating migration process from SQLite to Centralized database')
            migrate = Migrate(sqlite_db_file, glance.db.get_api())
            migrate.migrate()