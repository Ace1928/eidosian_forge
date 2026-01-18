import hashlib
import json
from unittest import mock
import uuid
import http.client
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_db import exception as oslo_db_exception
from testtools import matchers
import urllib
from keystone.api import ec2tokens
from keystone.common import provider_api
from keystone.common import utils
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone import oauth1
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
def test_delete_credential_retries_on_deadlock(self):
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
    try:
        PROVIDERS.credential_api.delete_credentials_for_user(user_id=self.user['id'])
    finally:
        if side_effect.patched:
            patcher.stop()
    self.assertEqual(sql_delete_mock.call_count, 2)