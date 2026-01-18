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
def read_domain_configs_from_files(self):
    """Read configs from file(s) and load into database.

        The command line parameters have already been parsed and the CONF
        command option will have been set. It is either set to the name of an
        explicit domain, or it's None to indicate that we want all domain
        config files.

        """
    domain_name = CONF.command.domain_name
    conf_dir = CONF.identity.domain_config_dir
    if not os.path.exists(conf_dir):
        print(_('Unable to locate domain config directory: %s') % conf_dir)
        raise ValueError
    if domain_name:
        fname = DOMAIN_CONF_FHEAD + domain_name + DOMAIN_CONF_FTAIL
        if not self._upload_config_to_database(os.path.join(conf_dir, fname), domain_name):
            return False
        return True
    success_cnt = 0
    failure_cnt = 0
    for filename, domain_name in self._domain_config_finder(conf_dir):
        if self._upload_config_to_database(filename, domain_name):
            success_cnt += 1
            LOG.info('Successfully uploaded domain config %r', filename)
        else:
            failure_cnt += 1
    if success_cnt == 0:
        LOG.warning('No domain configs uploaded from %r', conf_dir)
    if failure_cnt:
        return False
    return True