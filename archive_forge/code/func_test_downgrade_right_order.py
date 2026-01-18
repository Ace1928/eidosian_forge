from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
def test_downgrade_right_order(self):
    results = self.migration_manager.downgrade(None)
    self.assertEqual([100, 0], results)