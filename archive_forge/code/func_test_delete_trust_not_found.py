import datetime
import uuid
from oslo_utils import timeutils
from keystone.common import provider_api
from keystone import exception
from keystone.tests.unit import core
def test_delete_trust_not_found(self):
    trust_id = uuid.uuid4().hex
    self.assertRaises(exception.TrustNotFound, PROVIDERS.trust_api.delete_trust, trust_id)