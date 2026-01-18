import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_cross_domain_implied_roles_authentication(self):
    user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainB['id'])
    projectA = unit.new_project_ref(domain_id=self.domainA['id'])
    PROVIDERS.resource_api.create_project(projectA['id'], projectA)
    self.put('/roles/%s/implies/%s' % (self.domainA_role1['id'], self.domainB_role['id']), expected_status=http.client.CREATED)
    PROVIDERS.assignment_api.create_grant(self.domainA_role1['id'], user_id=user['id'], project_id=projectA['id'])
    assignments = PROVIDERS.assignment_api.list_role_assignments(user_id=user['id'], effective=True)
    self.assertEqual([], assignments)
    auth_body = self.build_authentication_request(user_id=user['id'], password=user['password'], project_id=projectA['id'])
    self.post('/auth/tokens', body=auth_body, expected_status=http.client.UNAUTHORIZED)