import uuid
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import filtering
@unit.skip_if_cache_disabled('identity')
def test_cache_layer_group_crud(self):
    group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group = PROVIDERS.identity_api.create_group(group)
    group_ref = PROVIDERS.identity_api.get_group(group['id'])
    domain_id, driver, entity_id = PROVIDERS.identity_api._get_domain_driver_and_entity_id(group['id'])
    driver.delete_group(entity_id)
    self.assertEqual(group_ref, PROVIDERS.identity_api.get_group(group['id']))
    PROVIDERS.identity_api.get_group.invalidate(PROVIDERS.identity_api, group['id'])
    self.assertRaises(exception.GroupNotFound, PROVIDERS.identity_api.get_group, group['id'])
    group = unit.new_group_ref(domain_id=CONF.identity.default_domain_id)
    group = PROVIDERS.identity_api.create_group(group)
    PROVIDERS.identity_api.get_group(group['id'])
    group['name'] = uuid.uuid4().hex
    group_ref = PROVIDERS.identity_api.update_group(group['id'], group)
    self.assertLessEqual(PROVIDERS.identity_api.get_group(group['id']).items(), group_ref.items())