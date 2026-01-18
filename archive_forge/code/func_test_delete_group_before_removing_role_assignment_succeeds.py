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
def test_delete_group_before_removing_role_assignment_succeeds(self):
    self.config_fixture.config(group='cache', enabled=False)
    group = unit.new_group_ref(domain_id=self.domain_id)
    group_ref = PROVIDERS.identity_api.create_group(group)
    collection_url = '/projects/%(project_id)s/groups/%(group_id)s/roles' % {'project_id': self.project_id, 'group_id': group_ref['id']}
    member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
    self.put(member_url)
    self.head(member_url)
    self.get(member_url, expected_status=http.client.NO_CONTENT)
    PROVIDERS.identity_api.driver.delete_group(group_ref['id'])
    self.delete(member_url)