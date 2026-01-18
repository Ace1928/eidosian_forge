import json
from unittest import mock
import fixtures
from oslo_serialization import jsonutils
from requests_mock.contrib import fixture as rm_fixture
from urllib import parse as urlparse
from oslo_policy import _external
from oslo_policy import opts
from oslo_policy.tests import base
def test_accept_json(self):
    self.conf.set_override('remote_content_type', 'application/json', group='oslo_policy')
    self.requests_mock.post('http://example.com/target', text='True')
    check = _external.HttpCheck('http', '//example.com/%(name)s')
    target_dict = dict(name='target', spam='spammer')
    cred_dict = dict(user='user', roles=['a', 'b', 'c'])
    self.assertTrue(check(target_dict, cred_dict, self.enforcer))
    last_request = self.requests_mock.last_request
    self.assertEqual('application/json', last_request.headers['Content-Type'])
    self.assertEqual('POST', last_request.method)
    self.assertEqual(dict(rule=None, credentials=cred_dict, target=target_dict), json.loads(last_request.body.decode('utf-8')))