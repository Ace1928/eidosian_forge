import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
def test_list_domains_for_user_with_grants(self):
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    user = unit.new_user_ref(domain_id=domain['id'])
    user = PROVIDERS.identity_api.create_user(user)
    group1 = unit.new_group_ref(domain_id=domain['id'])
    group1 = PROVIDERS.identity_api.create_group(group1)
    group2 = unit.new_group_ref(domain_id=domain['id'])
    group2 = PROVIDERS.identity_api.create_group(group2)
    test_domain1 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(test_domain1['id'], test_domain1)
    test_domain2 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(test_domain2['id'], test_domain2)
    test_domain3 = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(test_domain3['id'], test_domain3)
    PROVIDERS.identity_api.add_user_to_group(user['id'], group1['id'])
    PROVIDERS.identity_api.add_user_to_group(user['id'], group2['id'])
    PROVIDERS.assignment_api.create_grant(user_id=user['id'], domain_id=test_domain1['id'], role_id=self.role_member['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group1['id'], domain_id=test_domain2['id'], role_id=self.role_admin['id'])
    PROVIDERS.assignment_api.create_grant(group_id=group2['id'], domain_id=test_domain3['id'], role_id=self.role_admin['id'])
    user_domains = PROVIDERS.assignment_api.list_domains_for_user(user['id'])
    self.assertThat(user_domains, matchers.HasLength(3))