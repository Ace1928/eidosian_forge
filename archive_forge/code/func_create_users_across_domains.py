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
def create_users_across_domains(self):
    """Create a set of users, each with a role on their own domain."""
    initial_mappings = len(mapping_sql.list_id_mappings())
    users = {}
    users['user0'] = unit.create_user(PROVIDERS.identity_api, self.domain_default['id'])
    PROVIDERS.assignment_api.create_grant(user_id=users['user0']['id'], domain_id=self.domain_default['id'], role_id=self.role_member['id'])
    for x in range(1, self.domain_count):
        users['user%s' % x] = unit.create_user(PROVIDERS.identity_api, self.domains['domain%s' % x]['id'])
        PROVIDERS.assignment_api.create_grant(user_id=users['user%s' % x]['id'], domain_id=self.domains['domain%s' % x]['id'], role_id=self.role_member['id'])
    self.assertEqual(initial_mappings + self.domain_specific_count, len(mapping_sql.list_id_mappings()))
    return users