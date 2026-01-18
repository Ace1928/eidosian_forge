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
def test_multiple_sql_specific_drivers_fails(self):
    self.config_fixture.config(group='identity', driver='ldap')
    self.config_fixture.config(group='assignment', driver='sql')
    self.load_backends()
    self.domain_count = 3
    self.setup_initial_domains()
    PROVIDERS.identity_api.list_users(domain_scope=CONF.identity.default_domain_id)
    self.assertIsNotNone(self.get_config(self.domains['domain1']['id']))
    self.assertRaises(exception.MultipleSQLDriversInConfig, PROVIDERS.identity_api.domain_configs._load_config_from_file, PROVIDERS.resource_api, [unit.TESTCONF + '/domain_configs_one_extra_sql/' + 'keystone.domain2.conf'], 'domain2')