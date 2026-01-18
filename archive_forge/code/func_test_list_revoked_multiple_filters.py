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
def test_list_revoked_multiple_filters(self):
    revocation_backend = sql.Revoke()
    first_token = _sample_blank_token()
    first_token['user_id'] = uuid.uuid4().hex
    first_token['project_id'] = uuid.uuid4().hex
    first_token['audit_id'] = provider.random_urlsafe_str()
    PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(user_id=first_token['user_id'], project_id=first_token['project_id'], audit_id=first_token['audit_id']))
    self._assertTokenRevoked(first_token)
    self.assertEqual(1, len(revocation_backend.list_events(token=first_token)))
    second_token = _sample_blank_token()
    self._assertTokenNotRevoked(second_token)
    self.assertEqual(0, len(revocation_backend.list_events(token=second_token)))
    third_token = _sample_blank_token()
    third_token['project_id'] = uuid.uuid4().hex
    self._assertTokenNotRevoked(third_token)
    self.assertEqual(0, len(revocation_backend.list_events(token=third_token)))
    fourth_token = _sample_blank_token()
    fourth_token['user_id'] = uuid.uuid4().hex
    fourth_token['project_id'] = uuid.uuid4().hex
    fourth_token['audit_id'] = provider.random_urlsafe_str()
    PROVIDERS.revoke_api.revoke(revoke_model.RevokeEvent(project_id=fourth_token['project_id'], audit_id=fourth_token['audit_id']))
    self._assertTokenRevoked(fourth_token)
    self.assertEqual(1, len(revocation_backend.list_events(token=fourth_token)))