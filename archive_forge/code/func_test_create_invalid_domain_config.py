import copy
from unittest import mock
import uuid
from oslo_config import cfg
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
def test_create_invalid_domain_config(self):
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.create_config, self.domain['id'], {})
    config = {uuid.uuid4().hex: uuid.uuid4().hex}
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.create_config, self.domain['id'], config)
    config = {uuid.uuid4().hex: {uuid.uuid4().hex: uuid.uuid4().hex}}
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.create_config, self.domain['id'], config)
    config = {'ldap': {uuid.uuid4().hex: uuid.uuid4().hex}}
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.create_config, self.domain['id'], config)
    config = {'identity': {'user_tree_dn': uuid.uuid4().hex}}
    self.assertRaises(exception.InvalidDomainConfig, PROVIDERS.domain_config_api.create_config, self.domain['id'], config)