import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_proxy_none(self):
    os.environ['HTTP_PROXY'] = 'random_proxy'
    os.environ['HTTPS_PROXY'] = 'random_proxy_2'
    t_default = transport.Transport(endpoint='https://example.com', server_cert_validation='validate', username='test', password='test', auth_method='basic', proxy=None)
    t_default.build_session()
    self.assertEqual({'no_proxy': '*'}, t_default.session.proxies)