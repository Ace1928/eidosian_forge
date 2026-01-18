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
@mock.patch('os.path.exists')
@mock.patch('os.path.join', new=mock.MagicMock())
def test_migrate_if_required_db_not_found(self, mock_exists):
    mock_exists.return_value = False
    self.config(flavor='keystone+cache', group='paste_deploy')
    self.config(image_cache_driver='centralized_db')
    with mock.patch.object(sqlite_migration, 'LOG') as mock_log:
        sqlite_migration.migrate_if_required()
        mock_log.debug.assert_called_once_with('SQLite caching database not located, skipping migration')