import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def test_consume_use(self):
    trust_data = self.create_sample_trust(uuid.uuid4().hex, remaining_uses=2)
    PROVIDERS.trust_api.consume_use(trust_data['id'])
    t = PROVIDERS.trust_api.get_trust(trust_data['id'])
    self.assertEqual(1, t['remaining_uses'])
    PROVIDERS.trust_api.consume_use(trust_data['id'])
    self.assertRaises(exception.TrustNotFound, PROVIDERS.trust_api.get_trust, trust_data['id'])