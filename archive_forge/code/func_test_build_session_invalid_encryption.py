import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_invalid_encryption(self):
    with self.assertRaises(WinRMError) as exc:
        transport.Transport(endpoint='Endpoint', server_cert_validation='validate', username='test', password='test', auth_method='basic', message_encryption='invalid_value')
    self.assertEqual("invalid message_encryption arg: invalid_value. Should be 'auto', 'always', or 'never'", str(exc.exception))