import copy
from unittest import mock
import uuid
import fixtures
import http.client
import ldap
from oslo_log import versionutils
import pkg_resources
from testtools import matchers
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import identity
from keystone.identity.backends import ldap as ldap_identity
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends import sql as sql_identity
from keystone.identity.mapping_backends import mapping as map
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import identity_mapping as mapping_sql
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.ksfixtures import ldapdb
from keystone.tests.unit.resource import test_backends as resource_tests
def test_delete_domain_clears_sql_registration(self):
    """Ensure registration is deleted when a domain is deleted."""
    domain = unit.new_domain_ref()
    domain = PROVIDERS.resource_api.create_domain(domain['id'], domain)
    new_config = {'identity': {'driver': 'sql'}}
    PROVIDERS.domain_config_api.create_config(domain['id'], new_config)
    PROVIDERS.identity_api.domain_configs.get_domain_conf(domain['id'])
    PROVIDERS.domain_config_api.create_config(self.domains['domain1']['id'], new_config)
    self.assertRaises(exception.MultipleSQLDriversInConfig, PROVIDERS.identity_api.domain_configs.get_domain_conf, self.domains['domain1']['id'])
    PROVIDERS.domain_config_api.delete_config(self.domains['domain1']['id'])
    domain['enabled'] = False
    PROVIDERS.resource_api.update_domain(domain['id'], domain)
    PROVIDERS.resource_api.delete_domain(domain['id'])
    PROVIDERS.domain_config_api.create_config(self.domains['domain1']['id'], new_config)
    PROVIDERS.identity_api.domain_configs.get_domain_conf(self.domains['domain1']['id'])