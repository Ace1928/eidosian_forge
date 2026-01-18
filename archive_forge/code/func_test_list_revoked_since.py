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
def test_list_revoked_since(self):
    revocation_backend = sql.Revoke()
    token = _sample_blank_token()
    PROVIDERS.revoke_api.revoke_by_user(user_id=None)
    PROVIDERS.revoke_api.revoke_by_user(user_id=None)
    self.assertEqual(2, len(revocation_backend.list_events(token=token)))
    future = timeutils.utcnow() + datetime.timedelta(seconds=1000)
    token['issued_at'] = future
    self.assertEqual(0, len(revocation_backend.list_events(token=token)))