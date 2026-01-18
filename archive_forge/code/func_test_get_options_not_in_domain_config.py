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
def test_get_options_not_in_domain_config(self):
    self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.get_config, self.domain['id'])
    config = {'ldap': {'url': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
    self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.get_config, self.domain['id'], group='identity')
    self.assertRaises(exception.DomainConfigNotFound, PROVIDERS.domain_config_api.get_config, self.domain['id'], group='ldap', option='user_tree_dn')