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
def test_no_overwrite_config(self):
    default_config = {'ldap': {'url': uuid.uuid4().hex}, 'identity': {'driver': 'ldap'}}
    PROVIDERS.domain_config_api.create_config(CONF.identity.default_domain_id, default_config)
    provider_api.ProviderAPIs._clear_registry_instances()
    with mock.patch('builtins.print') as mock_print:
        self.assertRaises(unit.UnexpectedExit, cli.DomainConfigUpload.main)
        file_name = 'keystone.%s.conf' % self.default_domain['name']
        error_msg = _('Domain: %(domain)s already has a configuration defined - ignoring file: %(file)s.') % {'domain': self.default_domain['name'], 'file': os.path.join(CONF.identity.domain_config_dir, file_name)}
        mock_print.assert_has_calls([mock.call(error_msg)])
    res = PROVIDERS.domain_config_api.get_config(CONF.identity.default_domain_id)
    self.assertEqual(default_config, res)