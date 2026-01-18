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
def test_cache_layer_get_user(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    PROVIDERS.identity_api.create_user(user)
    ref = PROVIDERS.identity_api.get_user_by_name(user['name'], user['domain_id'])
    PROVIDERS.identity_api.get_user(ref['id'])
    domain_id, driver, entity_id = PROVIDERS.identity_api._get_domain_driver_and_entity_id(ref['id'])
    driver.delete_user(entity_id)
    self.assertDictEqual(ref, PROVIDERS.identity_api.get_user(ref['id']))
    PROVIDERS.identity_api.get_user.invalidate(PROVIDERS.identity_api, ref['id'])
    self.assertRaises(exception.UserNotFound, PROVIDERS.identity_api.get_user, ref['id'])
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    user = PROVIDERS.identity_api.create_user(user)
    ref = PROVIDERS.identity_api.get_user_by_name(user['name'], user['domain_id'])
    user['description'] = uuid.uuid4().hex
    PROVIDERS.identity_api.get_user(ref['id'])
    user_updated = PROVIDERS.identity_api.update_user(ref['id'], user)
    self.assertLessEqual(PROVIDERS.identity_api.get_user(ref['id']).items(), user_updated.items())
    self.assertLessEqual(PROVIDERS.identity_api.get_user_by_name(ref['name'], ref['domain_id']).items(), user_updated.items())