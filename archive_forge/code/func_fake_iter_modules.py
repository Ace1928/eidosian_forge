from unittest import mock
from glance.db.sqlalchemy.alembic_migrations import data_migrations
from glance.tests import utils as test_utils
def fake_iter_modules(blah):
    yield ('blah', 'zebra01', 'blah')
    yield ('blah', 'zebra02', 'blah')
    yield ('blah', 'yellow01', 'blah')
    yield ('blah', 'xray01', 'blah')
    yield ('blah', 'xray02', 'blah')