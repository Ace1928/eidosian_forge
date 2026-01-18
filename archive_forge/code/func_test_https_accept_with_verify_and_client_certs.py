import json
from unittest import mock
import fixtures
from oslo_serialization import jsonutils
from requests_mock.contrib import fixture as rm_fixture
from urllib import parse as urlparse
from oslo_policy import _external
from oslo_policy import opts
from oslo_policy.tests import base
def test_https_accept_with_verify_and_client_certs(self):
    self.conf.set_override('remote_ssl_verify_server_crt', True, group='oslo_policy')
    self.conf.set_override('remote_ssl_ca_crt_file', 'ca.crt', group='oslo_policy')
    self.conf.set_override('remote_ssl_client_key_file', 'client.key', group='oslo_policy')
    self.conf.set_override('remote_ssl_client_crt_file', 'client.crt', group='oslo_policy')
    self.requests_mock.post('https://example.com/target', text='True')
    check = _external.HttpsCheck('https', '//example.com/%(name)s')
    target_dict = dict(name='target', spam='spammer')
    cred_dict = dict(user='user', roles=['a', 'b', 'c'])
    with mock.patch('os.path.exists') as path_exists:
        with mock.patch('os.access') as os_access:
            path_exists.return_value = True
            os_access.return_value = True
            self.assertTrue(check(target_dict, cred_dict, self.enforcer))
    last_request = self.requests_mock.last_request
    self.assertEqual('ca.crt', last_request.verify)
    self.assertEqual(('client.crt', 'client.key'), last_request.cert)
    self.assertEqual('POST', last_request.method)
    self.assertEqual(dict(rule=None, target=target_dict, credentials=cred_dict), self.decode_post_data(last_request.body))