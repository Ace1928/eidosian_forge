from unittest import mock
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests import utils as test_utils
@mock.patch('glance.db.sqlalchemy.alembic_migrations.data_migrations._find_migration_modules')
def test_has_pending_migrations_one_migration_with_pending(self, mock_find):
    mock_migration1 = mock.Mock()
    mock_migration1.has_migrations.return_value = True
    mock_find.return_value = [mock_migration1]
    self.assertTrue(data_migrations.has_pending_migrations(mock.Mock()))