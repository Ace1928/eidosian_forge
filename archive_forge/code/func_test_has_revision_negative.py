from unittest import mock
import alembic
import sqlalchemy
from oslo_db import exception
from oslo_db.sqlalchemy.migration_cli import ext_alembic
from oslo_db.sqlalchemy.migration_cli import manager
from oslo_db.tests import base as test_base
def test_has_revision_negative(self, command):
    with mock.patch('oslo_db.sqlalchemy.migration_cli.ext_alembic.alembic_script') as mocked:
        mocked.ScriptDirectory().get_revision.side_effect = alembic.util.CommandError
        self.alembic.config.get_main_option = mock.Mock()
        self.assertIs(False, self.alembic.has_revision('test'))
        self.alembic.config.get_main_option.assert_called_once_with('script_location')
        mocked.ScriptDirectory().get_revision.assert_called_once_with('test')