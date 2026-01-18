import json
from unittest import mock
import fixtures
from oslo_serialization import jsonutils
from requests_mock.contrib import fixture as rm_fixture
from urllib import parse as urlparse
from oslo_policy import _external
from oslo_policy import opts
from oslo_policy.tests import base
def test_https_with_strings_in_target(self):
    self.requests_mock.post('https://example.com/target', text='True')
    check = _external.HttpsCheck('https', '//example.com/%(name)s')
    target = {'a': 'some_string', 'name': 'target', 'b': 'test data'}
    self.assertTrue(check(target, dict(user='user', roles=['a', 'b', 'c']), self.enforcer))