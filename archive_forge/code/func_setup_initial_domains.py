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
def setup_initial_domains(self):

    def create_domain(domain):
        return PROVIDERS.resource_api.create_domain(domain['id'], domain)
    PROVIDERS.resource_api.create_domain(default_fixtures.ROOT_DOMAIN['id'], default_fixtures.ROOT_DOMAIN)
    self.domains = {}
    self.addCleanup(self.cleanup_domains)
    for x in range(1, self.domain_count):
        domain = 'domain%s' % x
        self.domains[domain] = create_domain({'id': uuid.uuid4().hex, 'name': domain})
    self.default_domain = unit.new_domain_ref(description=u'The default domain', id=CONF.identity.default_domain_id, name=u'Default')
    self.domains['domain_default'] = create_domain(self.default_domain)