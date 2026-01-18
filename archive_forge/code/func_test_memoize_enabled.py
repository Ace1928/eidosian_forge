import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
@mock.patch('os.path.exists', return_value=True)
def test_memoize_enabled(self, exists):
    netutils._IS_IPV6_ENABLED = None
    with mock.patch('builtins.open', return_value=mock_file_content('0')) as mock_open:
        enabled = netutils.is_ipv6_enabled()
        self.assertTrue(mock_open.called)
        self.assertTrue(netutils._IS_IPV6_ENABLED)
        self.assertTrue(enabled)
    with mock.patch('builtins.open', side_effect=AssertionError('should not be called')):
        enabled = netutils.is_ipv6_enabled()
        self.assertTrue(enabled)