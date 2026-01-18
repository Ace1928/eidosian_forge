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
def test_authenticate_requires_simple_bind(self):
    user = self.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    role_member = unit.new_role_ref()
    PROVIDERS.role_api.create_role(role_member['id'], role_member)
    PROVIDERS.assignment_api.add_role_to_user_and_project(user['id'], self.project_baz['id'], role_member['id'])
    driver = PROVIDERS.identity_api._select_identity_driver(user['domain_id'])
    driver.user.LDAP_USER = None
    driver.user.LDAP_PASSWORD = None
    with self.make_request():
        self.assertRaises(AssertionError, PROVIDERS.identity_api.authenticate, user_id=user['id'], password=None)