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
def test_list_groups_by_name_and_with_filter(self):
    domain = self._get_domain_fixture()
    group_names = []
    numgroups = 3
    for _ in range(numgroups):
        group = unit.new_group_ref(domain_id=domain['id'])
        group = PROVIDERS.identity_api.create_group(group)
        group_names.append(group['name'])
    groups = PROVIDERS.identity_api.list_groups(domain_scope=self._set_domain_scope(domain['id']))
    self.assertEqual(numgroups, len(groups))
    driver = PROVIDERS.identity_api._select_identity_driver(domain['id'])
    driver.group.ldap_filter = '(|(ou=%s)(ou=%s))' % tuple(group_names[:2])
    groups = PROVIDERS.identity_api.list_groups(domain_scope=self._set_domain_scope(domain['id']))
    self.assertEqual(2, len(groups))
    hints = driver_hints.Hints()
    hints.add_filter('name', group_names[2])
    groups = PROVIDERS.identity_api.list_groups(domain_scope=self._set_domain_scope(domain['id']), hints=hints)
    self.assertEqual(0, len(groups))