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
def test_get_and_list_domain_specific_roles(self):
    r = self.get('/roles/%s' % self.domainA_role1['id'])
    self.assertValidRoleResponse(r, self.domainA_role1)
    r = self.get('/roles')
    self.assertValidRoleListResponse(r, expected_length=self.existing_global_roles)
    self.assertRoleInListResponse(r, self.global_role1)
    self.assertRoleInListResponse(r, self.global_role2)
    self.assertRoleNotInListResponse(r, self.domainA_role1)
    self.assertRoleNotInListResponse(r, self.domainA_role2)
    self.assertRoleNotInListResponse(r, self.domainB_role)
    r = self.get('/roles?domain_id=%s' % self.domainA['id'])
    self.assertValidRoleListResponse(r, expected_length=2)
    self.assertRoleInListResponse(r, self.domainA_role1)
    self.assertRoleInListResponse(r, self.domainA_role2)