from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
def test_manager_update(self):
    self.migration_manager.upgrade('head')
    self.ext.obj.upgrade.assert_called_once_with('head')