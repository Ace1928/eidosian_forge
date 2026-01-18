import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_cert_override_2(self):
    os.environ['CURL_CA_BUNDLE'] = 'path_to_CURL_CA_CERT'
    t_default = transport.Transport(endpoint='https://example.com', server_cert_validation='validate', username='test', password='test', auth_method='basic', ca_trust_path='overridepath')
    t_default.build_session()
    self.assertEqual('overridepath', t_default.session.verify)