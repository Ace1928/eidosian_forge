import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_cert_validate_default(self):
    t_default = transport.Transport(endpoint='https://example.com', username='test', password='test', auth_method='basic')
    t_default.build_session()
    self.assertEqual(True, t_default.session.verify)