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
@mock.patch.object(timeutils, 'utcnow')
def test_expired_events_are_removed(self, mock_utcnow):

    def _sample_token_values():
        token = _sample_blank_token()
        token['expires_at'] = utils.isotime(_future_time(), subsecond=True)
        return token
    now = datetime.datetime.utcnow()
    now_plus_2h = now + datetime.timedelta(hours=2)
    mock_utcnow.return_value = now
    token_values = _sample_token_values()
    audit_chain_id = uuid.uuid4().hex
    PROVIDERS.revoke_api.revoke_by_audit_chain_id(audit_chain_id)
    token_values['audit_chain_id'] = audit_chain_id
    self.assertRaises(exception.TokenNotFound, PROVIDERS.revoke_api.check_token, token_values)
    mock_utcnow.return_value = now_plus_2h
    PROVIDERS.revoke_api.revoke_by_audit_chain_id(audit_chain_id)
    self.assertRaises(exception.TokenNotFound, PROVIDERS.revoke_api.check_token, token_values)