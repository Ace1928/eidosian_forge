import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_proxy_defined_and_env(self):
    os.environ['HTTPS_PROXY'] = 'random_proxy'
    t_default = transport.Transport(endpoint='https://example.com', server_cert_validation='validate', username='test', password='test', auth_method='basic', proxy='test_proxy')
    t_default.build_session()
    self.assertEqual({'http': 'test_proxy', 'https': 'test_proxy'}, t_default.session.proxies)