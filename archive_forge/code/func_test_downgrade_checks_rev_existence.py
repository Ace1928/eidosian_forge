from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
def test_downgrade_checks_rev_existence(self):
    self.first_ext.obj.has_revision.return_value = False
    self.second_ext.obj.has_revision.return_value = False
    self.assertRaises(exception.DBMigrationError, self.migration_manager.downgrade, 100)
    self.assertEqual([100, 0], self.migration_manager.downgrade(None))
    self.first_ext.obj.has_revision.return_value = True
    self.assertEqual([100, 0], self.migration_manager.downgrade(200))
    self.assertEqual([100, 0], self.migration_manager.downgrade(None))
    self.assertEqual([100, 0], self.migration_manager.downgrade('base'))