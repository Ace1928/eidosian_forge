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
def check_bootstrap_new_branch(branch, version_path, addn_kwargs):
    """Bootstrap a new migration branch if it does not exist."""
    addn_kwargs['version_path'] = version_path
    addn_kwargs['head'] = f'{branch}@head'
    if not os.path.exists(version_path):
        fileutils.ensure_tree(version_path, mode=493)