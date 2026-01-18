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
def test_cross_domain_assignment_invalid(self):
    user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domainB['id'])
    projectB = unit.new_project_ref(domain_id=self.domainB['id'])
    PROVIDERS.resource_api.create_project(projectB['id'], projectB)
    self.assertRaises(exception.DomainSpecificRoleMismatch, PROVIDERS.assignment_api.create_grant, self.domainA_role1['id'], user_id=user['id'], project_id=projectB['id'])