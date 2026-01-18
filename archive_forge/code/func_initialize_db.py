from contextlib import contextmanager
import os
import sqlite3
import stat
import time
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import fileutils
from glance.common import exception
from glance.i18n import _, _LI, _LW
from glance.image_cache.drivers import base
from glance.image_cache.drivers import common
def initialize_db(self):
    db = CONF.image_cache_sqlite_db
    self.db_path = os.path.join(self.base_dir, db)
    lockutils.set_defaults(self.base_dir)

    @lockutils.synchronized('image_cache_db_init', external=True)
    def create_db():
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False, factory=common.SqliteConnection)
            conn.executescript('\n                    CREATE TABLE IF NOT EXISTS cached_images (\n                        image_id TEXT PRIMARY KEY,\n                        last_accessed REAL DEFAULT 0.0,\n                        last_modified REAL DEFAULT 0.0,\n                        size INTEGER DEFAULT 0,\n                        hits INTEGER DEFAULT 0,\n                        checksum TEXT\n                    );\n                ')
            conn.close()
        except sqlite3.DatabaseError as e:
            msg = _('Failed to initialize the image cache database. Got error: %s') % e
            LOG.error(msg)
            raise exception.BadDriverConfiguration(driver_name='sqlite', reason=msg)
    create_db()