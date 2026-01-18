from unittest import mock
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests import utils as test_utils
@mock.patch('glance.db.migration.CURRENT_RELEASE', 'zebra')
@mock.patch('importlib.import_module')
@mock.patch('pkgutil.iter_modules')
def test_migrate(self, mock_iter, mock_import):

    def fake_iter_modules(blah):
        yield ('blah', 'zebra01', 'blah')
        yield ('blah', 'zebra02', 'blah')
        yield ('blah', 'yellow01', 'blah')
        yield ('blah', 'xray01', 'blah')
        yield ('blah', 'xray02', 'blah')
    mock_iter.side_effect = fake_iter_modules
    zebra1 = mock.Mock()
    zebra1.has_migrations.return_value = True
    zebra1.migrate.return_value = 100
    zebra2 = mock.Mock()
    zebra2.has_migrations.return_value = True
    zebra2.migrate.return_value = 50
    fake_imported_modules = [zebra1, zebra2]
    mock_import.side_effect = fake_imported_modules
    engine = mock.Mock()
    actual = data_migrations.migrate(engine, 'zebra')
    self.assertEqual(150, actual)
    zebra1.has_migrations.assert_called_once_with(engine)
    zebra1.migrate.assert_called_once_with(engine)
    zebra2.has_migrations.assert_called_once_with(engine)
    zebra2.migrate.assert_called_once_with(engine)