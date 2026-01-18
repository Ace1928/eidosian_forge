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
def test_config_registration(self):
    type = uuid.uuid4().hex
    PROVIDERS.domain_config_api.obtain_registration(self.domain['id'], type)
    PROVIDERS.domain_config_api.release_registration(self.domain['id'], type=type)
    PROVIDERS.domain_config_api.obtain_registration(self.domain['id'], type)
    self.assertFalse(PROVIDERS.domain_config_api.obtain_registration(self.domain['id'], type))
    self.assertEqual(self.domain['id'], PROVIDERS.domain_config_api.read_registration(type))
    domain2 = {'id': uuid.uuid4().hex, 'name': uuid.uuid4().hex}
    PROVIDERS.resource_api.create_domain(domain2['id'], domain2)
    PROVIDERS.domain_config_api.release_registration(domain2['id'], type=type)
    PROVIDERS.domain_config_api.release_registration(self.domain['id'], type=type)
    self.assertRaises(exception.ConfigRegistrationNotFound, PROVIDERS.domain_config_api.read_registration, type)
    type2 = uuid.uuid4().hex
    PROVIDERS.domain_config_api.obtain_registration(self.domain['id'], type)
    PROVIDERS.domain_config_api.obtain_registration(self.domain['id'], type2)
    PROVIDERS.domain_config_api.release_registration(self.domain['id'])
    self.assertRaises(exception.ConfigRegistrationNotFound, PROVIDERS.domain_config_api.read_registration, type)
    self.assertRaises(exception.ConfigRegistrationNotFound, PROVIDERS.domain_config_api.read_registration, type2)