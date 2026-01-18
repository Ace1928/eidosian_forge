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
def test_list_revoked_user(self):
    revocation_backend = sql.Revoke()
    first_token = _sample_blank_token()
    first_token['user_id'] = uuid.uuid4().hex
    PROVIDERS.revoke_api.revoke_by_user(user_id=first_token['user_id'])
    self._assertTokenRevoked(first_token)
    self.assertEqual(1, len(revocation_backend.list_events(token=first_token)))
    second_token = _sample_blank_token()
    second_token['user_id'] = uuid.uuid4().hex
    PROVIDERS.revoke_api.revoke_by_user(user_id=second_token['user_id'])
    self._assertTokenRevoked(second_token)
    self.assertEqual(1, len(revocation_backend.list_events(token=second_token)))
    third_token = _sample_blank_token()
    third_token['user_id'] = uuid.uuid4().hex
    self._assertTokenNotRevoked(third_token)
    self.assertEqual(0, len(revocation_backend.list_events(token=third_token)))
    fourth_token = _sample_blank_token()
    fourth_token['user_id'] = None
    self._assertTokenNotRevoked(fourth_token)
    self.assertEqual(0, len(revocation_backend.list_events(token=fourth_token)))