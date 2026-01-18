from unittest import mock
import webob.exc
from glance.api.v2 import policy
from glance.common import exception
from glance.tests import utils
def test_get_members(self):
    self.policy.get_members()
    expected_calls = [mock.call(self.context, 'get_image', mock.ANY), mock.call(self.context, 'get_members', mock.ANY)]
    self.enforcer.enforce.assert_has_calls(expected_calls)