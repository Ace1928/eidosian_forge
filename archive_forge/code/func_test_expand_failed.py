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
@mock.patch('glance.db.sqlalchemy.alembic_migrations.get_current_alembic_heads')
@mock.patch('glance.db.sqlalchemy.alembic_migrations.get_alembic_branch_head')
@mock.patch.object(manage.DbCommands, '_validate_engine')
@mock.patch.object(manage.DbCommands, '_sync')
def test_expand_failed(self, mock_sync, mock_validate_engine, mock_get_alembic_branch_head, mock_get_current_alembic_heads):
    engine = mock_validate_engine.return_value
    engine.engine.name = 'mysql'
    mock_get_current_alembic_heads.side_effect = ['ocata_contract01', 'test']
    mock_get_alembic_branch_head.side_effect = ['pike_expand01', 'pike_contract01']
    exit = self.assertRaises(SystemExit, self.db.expand)
    mock_sync.assert_called_once_with(version='pike_expand01')
    self.assertIn('Database expansion failed. Database expansion should have brought the database version up to "pike_expand01" revision. But, current revisions are: test ', exit.code)