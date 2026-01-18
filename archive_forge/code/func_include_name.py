from logging.config import fileConfig
from alembic import context
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from keystone.common.sql import core
from keystone.common.sql.migrations import autogen
def include_name(name, type_, parent_names):
    """Determine which tables or columns to skip.

    This is used where we have migrations that are out-of-sync with the models.
    """
    REMOVED_TABLES = ('token',)
    if type_ == 'table':
        return name not in REMOVED_TABLES
    return True