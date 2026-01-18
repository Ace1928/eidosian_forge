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
def read_assertion(self, path):
    self.assertion_pathname = path
    try:
        with open(path) as file:
            self.assertion = file.read().strip()
    except IOError as e:
        raise SystemExit(_('Error while opening file %(path)s: %(err)s') % {'path': path, 'err': e})
    LOG.debug('Assertions loaded: [%s].', self.assertion)