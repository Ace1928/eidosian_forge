import json
from unittest import mock
import fixtures
from oslo_serialization import jsonutils
from requests_mock.contrib import fixture as rm_fixture
from urllib import parse as urlparse
from oslo_policy import _external
from oslo_policy import opts
from oslo_policy.tests import base
def test_https_accept_with_verify(self):
    self.conf.set_override('remote_ssl_verify_server_crt', True, group='oslo_policy')
    self.conf.set_override('remote_ssl_ca_crt_file', None, group='oslo_policy')
    self.requests_mock.post('https://example.com/target', text='True')
    check = _external.HttpsCheck('https', '//example.com/%(name)s')
    target_dict = dict(name='target', spam='spammer')
    cred_dict = dict(user='user', roles=['a', 'b', 'c'])
    self.assertTrue(check(target_dict, cred_dict, self.enforcer))
    last_request = self.requests_mock.last_request
    self.assertEqual(True, last_request.verify)
    self.assertEqual('POST', last_request.method)
    self.assertEqual(dict(rule=None, target=target_dict, credentials=cred_dict), self.decode_post_data(last_request.body))