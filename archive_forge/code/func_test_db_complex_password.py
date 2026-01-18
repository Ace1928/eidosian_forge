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
def test_db_complex_password(self):
    engine = mock.Mock()
    engine.url = sqlalchemy_make_url('mysql+pymysql://username:pw@%/!#$()@host:1234/dbname')
    alembic_config = alembic_migrations.get_alembic_config(engine)
    self.assertEqual(str(engine.url), alembic_config.get_main_option('sqlalchemy.url'))