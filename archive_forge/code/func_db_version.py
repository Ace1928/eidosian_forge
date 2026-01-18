import os
from alembic import command as alembic_api
from alembic import config as alembic_config
from alembic import migration as alembic_migration
from oslo_log import log as logging
import sqlalchemy as sa
from heat.db import api as db_api
def db_version():
    """Get database version."""
    engine = db_api.get_engine()
    with engine.connect() as connection:
        m_context = alembic_migration.MigrationContext.configure(connection)
        return m_context.get_current_revision()