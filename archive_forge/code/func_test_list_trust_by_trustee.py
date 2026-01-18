import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def test_list_trust_by_trustee(self):
    for i in range(3):
        self.create_sample_trust(uuid.uuid4().hex)
    trusts = PROVIDERS.trust_api.list_trusts_for_trustee(self.trustee['id'])
    self.assertEqual(3, len(trusts))
    self.assertEqual(trusts[0]['trustee_user_id'], self.trustee['id'])
    trusts = PROVIDERS.trust_api.list_trusts_for_trustee(self.trustor['id'])
    self.assertEqual(0, len(trusts))