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
@unit.skip_if_cache_disabled('assignment')
def test_delete_grant_from_user_and_domain_invalidates_cache(self):
    new_domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
    collection_url = '/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': new_domain['id'], 'user_id': self.user['id']}
    member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
    self.put(member_url)
    self.head(member_url)
    self.get(member_url, expected_status=http.client.NO_CONTENT)
    resp = self.get(collection_url)
    self.assertValidRoleListResponse(resp, ref=self.role, resource_url=collection_url)
    self.delete(member_url)
    resp = self.get(collection_url)
    self.assertListEqual(resp.json_body['roles'], [])