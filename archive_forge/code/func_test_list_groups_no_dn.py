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
def test_list_groups_no_dn(self):
    domain = self._get_domain_fixture()
    expected_group_ids = []
    numgroups = 3
    for _ in range(numgroups):
        group = unit.new_group_ref(domain_id=domain['id'])
        group = PROVIDERS.identity_api.create_group(group)
        expected_group_ids.append(group['id'])
    groups = PROVIDERS.identity_api.list_groups()
    self.assertEqual(numgroups, len(groups))
    group_ids = set((group['id'] for group in groups))
    for group_ref in groups:
        self.assertNotIn('dn', group_ref)
    self.assertEqual(set(expected_group_ids), group_ids)