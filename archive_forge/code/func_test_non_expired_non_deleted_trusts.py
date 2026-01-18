import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def test_non_expired_non_deleted_trusts(self):
    roles = [{'id': 'member'}, {'id': 'other'}, {'id': 'browser'}]
    trust_ref1 = core.new_trust_ref(self.user_foo['id'], self.user_two['id'], project_id=self.project_bar['id'])
    trust_ref1['expires_at'] = timeutils.utcnow() + datetime.timedelta(minutes=10)
    trust_ref2 = core.new_trust_ref(self.user_two['id'], self.user_two['id'], project_id=self.project_bar['id'])
    trust_ref2['expires_at'] = timeutils.utcnow() + datetime.timedelta(minutes=5)
    trust_ref3 = core.new_trust_ref(self.user_two['id'], self.user_foo['id'], project_id=self.project_bar['id'])
    trust_ref3['expires_at'] = None
    trust_data = PROVIDERS.trust_api.create_trust(trust_ref1['id'], trust_ref1, roles)
    self.assertIsNotNone(trust_data)
    trust_data = PROVIDERS.trust_api.create_trust(trust_ref2['id'], trust_ref2, roles)
    self.assertIsNotNone(trust_data)
    PROVIDERS.trust_api.delete_trust(trust_ref2['id'])
    trust_data = PROVIDERS.trust_api.create_trust(trust_ref3['id'], trust_ref3, roles)
    self.assertIsNotNone(trust_data)
    PROVIDERS.trust_api.flush_expired_and_soft_deleted_trusts(date=datetime.datetime.utcnow())
    trusts = self.trust_api.list_trusts()
    self.assertEqual(len(trusts), 2)