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
def test_get_domain_mapping_list_is_used(self):
    for i in range(5):
        unit.create_user(PROVIDERS.identity_api, domain_id=self.domains['domain1']['id'])
    with mock.patch.multiple(PROVIDERS.id_mapping_api, get_domain_mapping_list=mock.DEFAULT, get_id_mapping=mock.DEFAULT) as mocked:
        PROVIDERS.identity_api.list_users(domain_scope=self.domains['domain1']['id'])
        mocked['get_domain_mapping_list'].assert_called()
        mocked['get_id_mapping'].assert_not_called()