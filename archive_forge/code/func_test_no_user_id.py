import io
import re
import sys
from unittest import mock
import fixtures
from keystoneauth1 import fixture
from testtools import matchers
from zunclient import api_versions
from zunclient import exceptions
import zunclient.shell
from zunclient.tests.unit import utils
def test_no_user_id(self):
    required = 'You must provide a username via either --os-username or env[OS_USERNAME]'
    self.make_env(exclude='OS_USER_ID', fake_env=FAKE_ENV2)
    try:
        self.shell('service-list')
    except exceptions.CommandError as message:
        self.assertEqual(required, message.args[0])
    else:
        self.fail('CommandError not raised')