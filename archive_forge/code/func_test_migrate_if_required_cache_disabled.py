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
def test_migrate_if_required_cache_disabled(self):
    self.config(flavor='keystone', group='paste_deploy')
    self.config(image_cache_driver='centralized_db')
    self.assertFalse(sqlite_migration.migrate_if_required())