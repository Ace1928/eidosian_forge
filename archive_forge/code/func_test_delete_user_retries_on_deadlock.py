import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
import http.client
from oslo_db import exception as oslo_db_exception
from oslo_log import log
from testtools import matchers
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.identity.backends import base as identity_base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import mapping_fixtures
from keystone.tests.unit import test_v3
def test_delete_user_retries_on_deadlock(self):
    patcher = mock.patch('sqlalchemy.orm.query.Query.delete', autospec=True)

    class FakeDeadlock(object):

        def __init__(self, mock_patcher):
            self.deadlock_count = 2
            self.mock_patcher = mock_patcher
            self.patched = True

        def __call__(self, *args, **kwargs):
            if self.deadlock_count > 1:
                self.deadlock_count -= 1
            else:
                self.mock_patcher.stop()
                self.patched = False
            raise oslo_db_exception.DBDeadlock
    sql_delete_mock = patcher.start()
    side_effect = FakeDeadlock(patcher)
    sql_delete_mock.side_effect = side_effect
    user_ref = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain['id'])
    try:
        PROVIDERS.identity_api.delete_user(user_id=user_ref['id'])
    finally:
        if side_effect.patched:
            patcher.stop()
    call_count = sql_delete_mock.call_count
    delete_user_attempt_count = 2
    self.assertEqual(call_count, delete_user_attempt_count)