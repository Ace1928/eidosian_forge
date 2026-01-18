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
def test_list_since(self):
    PROVIDERS.revoke_api.revoke_by_user(user_id=1)
    PROVIDERS.revoke_api.revoke_by_user(user_id=2)
    past = timeutils.utcnow() - datetime.timedelta(seconds=1000)
    self.assertEqual(2, len(PROVIDERS.revoke_api.list_events(last_fetch=past)))
    future = timeutils.utcnow() + datetime.timedelta(seconds=1000)
    self.assertEqual(0, len(PROVIDERS.revoke_api.list_events(last_fetch=future)))