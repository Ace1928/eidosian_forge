import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def test_get_deleted_trust(self):
    new_id = uuid.uuid4().hex
    trust_data = self.create_sample_trust(new_id)
    self.assertIsNotNone(trust_data)
    self.assertIsNone(trust_data['deleted_at'])
    PROVIDERS.trust_api.delete_trust(new_id)
    self.assertRaises(exception.TrustNotFound, PROVIDERS.trust_api.get_trust, new_id)
    deleted_trust = PROVIDERS.trust_api.get_trust(trust_data['id'], deleted=True)
    self.assertEqual(trust_data['id'], deleted_trust['id'])
    self.assertIsNotNone(deleted_trust.get('deleted_at'))