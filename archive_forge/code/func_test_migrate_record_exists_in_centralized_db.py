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
def test_migrate_record_exists_in_centralized_db(self):
    self.create_db()
    self.initialize_fake_cache_details()
    with mock.patch.object(sqlite_migration, 'LOG') as mock_log:
        with mock.patch.object(self.db_api, 'is_image_cached_for_node') as mock_call:
            mock_call.return_value = True
            self.migrate.migrate()
        expected_calls = [mock.call('Adding local node reference %(node)s in centralized db', {'node': 'http://worker1.example.com'}), mock.call('Connecting to SQLite db %s', self.db), mock.call('Skipping migrating image %(uuid)s from SQLite to Centralized db for node %(node)s as it is present in Centralized db.', {'uuid': FAKE_IMAGE_1, 'node': 'http://worker1.example.com'})]
        mock_log.debug.assert_has_calls(expected_calls)