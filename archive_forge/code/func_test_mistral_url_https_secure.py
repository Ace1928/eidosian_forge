import os
import tempfile
from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from oslotest import base
import osprofiler.profiler
from mistralclient.api import client
@mock.patch('keystoneauth1.session.Session')
@mock.patch('mistralclient.api.httpclient.HTTPClient')
def test_mistral_url_https_secure(self, http_client_mock, session_mock):
    fd, cert_path = tempfile.mkstemp(suffix='.pem')
    keystone_client_instance = self.setup_keystone_mock(session_mock)
    expected_args = (MISTRAL_HTTPS_URL,)
    try:
        client.client(mistral_url=MISTRAL_HTTPS_URL, username='mistral', project_name='mistral', api_key='password', user_domain_name='Default', project_domain_name='Default', auth_url=AUTH_HTTP_URL_v3, cacert=cert_path, insecure=False)
    finally:
        os.close(fd)
        os.unlink(cert_path)
    self.assertTrue(http_client_mock.called)
    self.assertEqual(http_client_mock.call_args[0], expected_args)
    self.assertEqual(http_client_mock.call_args[1]['cacert'], cert_path)