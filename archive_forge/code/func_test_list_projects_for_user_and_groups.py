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
def test_list_projects_for_user_and_groups(self):
    domain = self._get_domain_fixture()
    user1 = self.new_user_ref(domain_id=domain['id'])
    user1 = PROVIDERS.identity_api.create_user(user1)
    group1 = unit.new_group_ref(domain_id=domain['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    PROVIDERS.identity_api.add_user_to_group(user1['id'], group1['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
    user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
    self.assertThat(user_projects, matchers.HasLength(1))
    PROVIDERS.assignment_api.delete_grant(user_id=user1['id'], project_id=self.project_bar['id'], role_id=self.role_member['id'])
    user_projects = PROVIDERS.assignment_api.list_projects_for_user(user1['id'])
    self.assertThat(user_projects, matchers.HasLength(1))