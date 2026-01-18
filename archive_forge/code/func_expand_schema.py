import os
from alembic import command as alembic_api
from alembic import config as alembic_config
from alembic import migration as alembic_migration
from alembic import script as alembic_script
from oslo_db import exception as db_exception
from oslo_log import log as logging
from oslo_utils import fileutils
from keystone.common import sql
import keystone.conf
def expand_schema(engine=None):
    """Expand the database schema ahead of data migration.

    This is run manually by the keystone-manage command before the first
    keystone node is migrated to the latest release.
    """
    _validate_upgrade_order(EXPAND_BRANCH, engine=engine)
    _db_sync(EXPAND_BRANCH, engine=engine)