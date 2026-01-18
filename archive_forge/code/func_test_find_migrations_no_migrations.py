from unittest import mock
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests import utils as test_utils
@mock.patch('pkgutil.iter_modules')
def test_find_migrations_no_migrations(self, mock_iter):

    def fake_iter_modules(blah):
        yield ('blah', 'zebra01', 'blah')
        yield ('blah', 'yellow01', 'blah')
        yield ('blah', 'xray01', 'blah')
        yield ('blah', 'wrinkle01', 'blah')
        yield ('blah', 'victor01', 'blah')
    mock_iter.side_effect = fake_iter_modules
    actual = data_migrations._find_migration_modules('umbrella')
    self.assertEqual(0, len(actual))
    self.assertEqual([], actual)