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
@mock.patch.object(ldap_identity.Identity, 'unset_default_project_id')
@mock.patch.object(sql_identity.Identity, 'unset_default_project_id')
def test_delete_project_unset_project_ids_for_all_backends(self, sql_mock, ldap_mock):
    ldap_mock.side_effect = exception.Forbidden
    project = unit.new_project_ref(domain_id=CONF.identity.default_domain_id)
    project = PROVIDERS.resource_api.create_project(project['id'], project)
    PROVIDERS.resource_api.delete_project(project['id'])
    ldap_mock.assert_called_with(project['id'])
    sql_mock.assert_called_with(project['id'])