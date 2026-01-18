import datetime
from unittest import mock
import uuid
from oslo_utils import timeutils
from testtools import matchers
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.models import revoke_model
from keystone.revoke.backends import sql
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_backend_sql
from keystone.token import provider
def test_by_domain_user(self):
    revocation_backend = sql.Revoke()
    user_id = uuid.uuid4().hex
    domain_id = uuid.uuid4().hex
    token_data = _sample_blank_token()
    token_data['user_id'] = user_id
    token_data['identity_domain_id'] = domain_id
    self._assertTokenNotRevoked(token_data)
    self.assertEqual(0, len(revocation_backend.list_events(token=token_data)))
    PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(domain_id=domain_id))
    self._assertTokenRevoked(token_data)
    self.assertEqual(1, len(revocation_backend.list_events(token=token_data)))