import copy
import datetime
import logging
import os
from unittest import mock
import uuid
import argparse
import configparser
import fixtures
import freezegun
import http.client
import oslo_config.fixture
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_upgradecheck import upgradecheck
from testtools import matchers
from keystone.cmd import cli
from keystone.cmd.doctor import caching
from keystone.cmd.doctor import credential
from keystone.cmd.doctor import database as doc_database
from keystone.cmd.doctor import debug
from keystone.cmd.doctor import federation
from keystone.cmd.doctor import ldap
from keystone.cmd.doctor import security_compliance
from keystone.cmd.doctor import tokens
from keystone.cmd.doctor import tokens_fernet
from keystone.cmd import status
from keystone.common import provider_api
from keystone.common.sql import upgrades
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.mapping_backends import mapping as identity_mapping
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.ksfixtures import policy
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import mapping_fixtures
def test_bootstrap_with_ambiguous_role_names(self):
    self._do_test_bootstrap(self.bootstrap)
    domain = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex}
    domain = PROVIDERS.resource_api.create_domain(domain['id'], domain)
    domain_roles = {}
    for name in ['admin', 'member', 'reader', 'service']:
        domain_role = {'domain_id': domain['id'], 'id': uuid.uuid4().hex, 'name': name}
        domain_roles[name] = PROVIDERS.role_api.create_role(domain_role['id'], domain_role)
        self._do_test_bootstrap(self.bootstrap)