import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_no_password(self):
    with self.assertRaises(InvalidCredentialsError) as exc:
        transport.Transport(endpoint='Endpoint', server_cert_validation='validate', username='test', auth_method='basic')
    self.assertEqual('auth method basic requires a password', str(exc.exception))