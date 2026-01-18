import argparse
import datetime
import os
import sys
import uuid
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_log import log
from oslo_serialization import jsonutils
import pbr.version
from keystone.cmd import bootstrap
from keystone.cmd import doctor
from keystone.cmd import idutils
from keystone.common import driver_hints
from keystone.common import fernet_utils
from keystone.common import jwt_utils
from keystone.common import sql
from keystone.common.sql import upgrades
from keystone.common import utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.federation import idp
from keystone.federation import utils as mapping_engine
from keystone.i18n import _
from keystone.server import backends
@classmethod
def check_db_sync_status(cls):
    status = 0
    try:
        expand_version = upgrades.get_db_version(branch='expand')
    except db_exception.DBMigrationError:
        LOG.info('Your database is not currently under version control or the database is already controlled. Your first step is to run `keystone-manage db_sync --expand`.')
        return 2
    try:
        contract_version = upgrades.get_db_version(branch='contract')
    except db_exception.DBMigrationError:
        contract_version = None
    heads = upgrades.get_current_heads()
    if upgrades.EXPAND_BRANCH not in heads or heads[upgrades.EXPAND_BRANCH] != expand_version:
        LOG.info('Your database is not up to date. Your first step is to run `keystone-manage db_sync --expand`.')
        status = 2
    elif upgrades.CONTRACT_BRANCH not in heads or heads[upgrades.CONTRACT_BRANCH] != contract_version:
        LOG.info('Expand version is ahead of contract. Your next step is to run `keystone-manage db_sync --contract`.')
        status = 4
    else:
        LOG.info('All db_sync commands are upgraded to the same version and up-to-date.')
    LOG.info('Current repository versions:\nExpand: %(expand)s (head: %(expand_head)s)\nContract: %(contract)s (head: %(contract_head)s)', {'expand': expand_version, 'expand_head': heads.get(upgrades.EXPAND_BRANCH), 'contract': contract_version, 'contract_head': heads.get(upgrades.CONTRACT_BRANCH)})
    return status