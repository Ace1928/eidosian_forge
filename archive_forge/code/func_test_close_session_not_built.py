import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
@mock.patch('requests.Session')
def test_close_session_not_built(self, mock_session):
    t_default = transport.Transport(endpoint='Endpoint', server_cert_validation='ignore', username='test', password='test', auth_method='basic')
    t_default.close_session()
    self.assertFalse(mock_session.return_value.close.called)
    self.assertIsNone(t_default.session)