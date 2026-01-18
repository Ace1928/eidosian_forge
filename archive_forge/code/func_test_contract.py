import io
from unittest import mock
import fixtures
from glance.cmd import manage
from glance.common import exception
from glance.db.sqlalchemy import alembic_migrations
from glance.db.sqlalchemy import api as db_api
from glance.db.sqlalchemy import metadata as db_metadata
from glance.tests import utils as test_utils
from sqlalchemy.engine.url import make_url as sqlalchemy_make_url
@mock.patch('glance.db.sqlalchemy.alembic_migrations.data_migrations.has_pending_migrations')
@mock.patch('glance.db.sqlalchemy.alembic_migrations.get_current_alembic_heads')
@mock.patch('glance.db.sqlalchemy.alembic_migrations.get_alembic_branch_head')
@mock.patch.object(manage.DbCommands, '_validate_engine')
@mock.patch.object(manage.DbCommands, '_sync')
def test_contract(self, mock_sync, mock_validate_engine, mock_get_alembic_branch_head, mock_get_current_alembic_heads, mock_has_pending_migrations):
    engine = mock_validate_engine.return_value
    engine.engine.name = 'mysql'
    mock_get_current_alembic_heads.side_effect = ['pike_expand01', 'pike_contract01']
    mock_get_alembic_branch_head.side_effect = ['pike_contract01', 'pike_expand01']
    mock_has_pending_migrations.return_value = False
    self.db.contract()
    mock_sync.assert_called_once_with(version='pike_contract01')