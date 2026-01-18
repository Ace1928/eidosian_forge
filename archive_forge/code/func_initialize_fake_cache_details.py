from contextlib import contextmanager
import os
import sqlite3
import tempfile
import time
from unittest import mock
import uuid
from oslo_config import cfg
from glance import sqlite_migration
import glance.tests.unit.utils as unit_test_utils
import glance.tests.utils as test_utils
def initialize_fake_cache_details(self):
    with self.get_db() as sq_db:
        filesize = 100
        now = time.time()
        sq_db.execute('INSERT INTO cached_images (image_id,\n                last_accessed, last_modified, hits, size)\n                VALUES (?, ?, ?, ?, ?)', (FAKE_IMAGE_1, now, now, 0, filesize))
        sq_db.commit()