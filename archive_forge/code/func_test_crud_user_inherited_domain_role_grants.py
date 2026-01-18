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
def test_crud_user_inherited_domain_role_grants(self):
    role_list = []
    for _ in range(2):
        role = unit.new_role_ref()
        PROVIDERS.role_api.create_role(role['id'], role)
        role_list.append(role)
    PROVIDERS.assignment_api.create_grant(role_list[1]['id'], user_id=self.user['id'], domain_id=self.domain_id)
    base_collection_url = '/OS-INHERIT/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': self.domain_id, 'user_id': self.user['id']}
    member_url = '%(collection_url)s/%(role_id)s/inherited_to_projects' % {'collection_url': base_collection_url, 'role_id': role_list[0]['id']}
    collection_url = base_collection_url + '/inherited_to_projects'
    self.put(member_url)
    self.head(member_url)
    self.get(member_url, expected_status=http.client.NO_CONTENT)
    r = self.get(collection_url)
    self.assertValidRoleListResponse(r, ref=role_list[0], resource_url=collection_url)
    self.delete(member_url)
    r = self.get(collection_url)
    self.assertValidRoleListResponse(r, expected_length=0, resource_url=collection_url)