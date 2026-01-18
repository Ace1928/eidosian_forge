from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
def test_downgrade_does_not_go_too_far(self):
    self.second_ext.obj.has_revision.return_value = True
    self.first_ext.obj.has_revision.return_value = False
    self.first_ext.obj.downgrade.side_effect = AssertionError('this method should not have been called')
    results = self.migration_manager.downgrade(100)
    self.assertEqual([100], results)