import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
@mock.patch.object(netutils, 'LOG', autospec=True)
def test_is_valid_ipv4(self, mock_log):
    expected_log = 'Converting in non strict mode is deprecated. You should pass strict=False if you want to preserve legacy behavior'
    self.assertTrue(netutils.is_valid_ipv4('42.42.42.42'))
    self.assertFalse(netutils.is_valid_ipv4('-1.11.11.11'))
    self.assertFalse(netutils.is_valid_ipv4(''))
    self.assertTrue(netutils.is_valid_ipv4('10'))
    mock_log.warning.assert_called_with(expected_log)
    mock_log.reset_mock()
    self.assertTrue(netutils.is_valid_ipv4('10.10'))
    mock_log.warning.assert_called_with(expected_log)
    mock_log.reset_mock()
    self.assertTrue(netutils.is_valid_ipv4('10.10.10'))
    mock_log.warning.assert_called_with(expected_log)
    mock_log.reset_mock()
    self.assertTrue(netutils.is_valid_ipv4('10.10.10.10'))
    mock_log.warning.assert_not_called()
    mock_log.reset_mock()
    self.assertFalse(netutils.is_valid_ipv4('10', strict=True))
    self.assertFalse(netutils.is_valid_ipv4('10.10', strict=True))
    self.assertFalse(netutils.is_valid_ipv4('10.10.10', strict=True))
    mock_log.warning.assert_not_called()
    mock_log.reset_mock()
    self.assertTrue(netutils.is_valid_ipv4('10', strict=False))
    self.assertTrue(netutils.is_valid_ipv4('10.10', strict=False))
    self.assertTrue(netutils.is_valid_ipv4('10.10.10', strict=False))
    mock_log.warning.assert_not_called()
    mock_log.reset_mock()