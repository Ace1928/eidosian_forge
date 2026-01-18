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
def normalize_assertion(self):

    def split(line, line_num):
        try:
            k, v = line.split(':', 1)
            return (k.strip(), v.strip())
        except ValueError:
            msg = _("assertion file %(pathname)s at line %(line_num)d expected 'key: value' but found '%(line)s' see help for file format")
            raise SystemExit(msg % {'pathname': self.assertion_pathname, 'line_num': line_num, 'line': line})
    assertion = self.assertion.splitlines()
    assertion_dict = {}
    prefix = CONF.command.prefix
    for line_num, line in enumerate(assertion, 1):
        line = line.strip()
        if line == '':
            continue
        k, v = split(line, line_num)
        if prefix:
            if k.startswith(prefix):
                assertion_dict[k] = v
        else:
            assertion_dict[k] = v
    self.assertion = assertion_dict