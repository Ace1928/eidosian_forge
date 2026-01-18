import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_cert_validate_default_env(self):
    os.environ['REQUESTS_CA_BUNDLE'] = 'path_to_REQUESTS_CA_CERT'
    t_default = transport.Transport(endpoint='https://example.com', username='test', password='test', auth_method='basic')
    t_default.build_session()
    self.assertEqual('path_to_REQUESTS_CA_CERT', t_default.session.verify)