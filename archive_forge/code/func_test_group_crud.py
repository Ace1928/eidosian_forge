import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
def test_group_crud(self):
    domain = unit.new_domain_ref()
    PROVIDERS.resource_api.create_domain(domain['id'], domain)
    group = unit.new_group_ref(domain_id=domain['id'])
    group = PROVIDERS.identity_api.create_group(group)
    group_ref = PROVIDERS.identity_api.get_group(group['id'])
    self.assertLessEqual(group.items(), group_ref.items())
    group['name'] = uuid.uuid4().hex
    PROVIDERS.identity_api.update_group(group['id'], group)
    group_ref = PROVIDERS.identity_api.get_group(group['id'])
    self.assertLessEqual(group.items(), group_ref.items())
    PROVIDERS.identity_api.delete_group(group['id'])
    self.assertRaises(exception.GroupNotFound, PROVIDERS.identity_api.get_group, group['id'])