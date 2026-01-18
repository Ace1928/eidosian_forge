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
def drop_db(self):
    if os.path.exists(self.db):
        os.remove(self.db)