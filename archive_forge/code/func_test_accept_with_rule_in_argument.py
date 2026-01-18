import json
from unittest import mock
import fixtures
from oslo_serialization import jsonutils
from requests_mock.contrib import fixture as rm_fixture
from urllib import parse as urlparse
from oslo_policy import _external
from oslo_policy import opts
from oslo_policy.tests import base
def test_accept_with_rule_in_argument(self):
    self.requests_mock.post('http://example.com/target', text='True')
    check = _external.HttpCheck('http', '//example.com/%(name)s')
    target_dict = dict(name='target', spam='spammer')
    cred_dict = dict(user='user', roles=['a', 'b', 'c'])
    current_rule = 'a_rule'
    self.assertTrue(check(target_dict, cred_dict, self.enforcer, current_rule))
    last_request = self.requests_mock.last_request
    self.assertEqual('POST', last_request.method)
    self.assertEqual(dict(target=target_dict, credentials=cred_dict, rule=current_rule), self.decode_post_data(last_request.body))