import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_invalid_auth(self):
    winrm_transport = transport.Transport(endpoint='Endpoint', server_cert_validation='validate', username='test', password='test', auth_method='invalid_value')
    with self.assertRaises(WinRMError) as exc:
        winrm_transport.build_session()
    self.assertEqual('unsupported auth method: invalid_value', str(exc.exception))