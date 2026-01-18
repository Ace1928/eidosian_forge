import os
from alembic import config as alembic_config
from alembic import migration as alembic_migration
from alembic import script as alembic_script
from sqlalchemy import MetaData, Table
from glance.db.sqlalchemy import api as db_api
def get_alembic_branch_head(branch):
    """Return head revision name for particular branch"""
    a_config = get_alembic_config()
    script = alembic_script.ScriptDirectory.from_config(a_config)
    return script.revision_map.get_current_head(branch)