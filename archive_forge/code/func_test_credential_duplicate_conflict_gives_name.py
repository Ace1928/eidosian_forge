import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_credential_duplicate_conflict_gives_name(self):
    user = unit.new_user_ref(domain_id=CONF.identity.default_domain_id)
    credential = unit.new_credential_ref(user_id=user['id'])
    PROVIDERS.credential_api.create_credential(credential['id'], credential)
    try:
        PROVIDERS.credential_api.create_credential(credential['id'], credential)
    except exception.Conflict as e:
        self.assertIn('Duplicate entry found with ID %s' % credential['id'], repr(e))
    else:
        self.fail('Create duplicate credential did not raise a conflict')