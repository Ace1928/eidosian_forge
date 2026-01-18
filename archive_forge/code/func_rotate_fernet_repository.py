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
def rotate_fernet_repository(cls, keystone_user_id, keystone_group_id, config_group=None):
    conf_group = getattr(CONF, config_group)
    futils = fernet_utils.FernetUtils(conf_group.key_repository, conf_group.max_active_keys, config_group)
    if futils.validate_key_repository(requires_write=True):
        futils.rotate_keys(keystone_user_id, keystone_group_id)