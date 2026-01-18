from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import smbutils
@ddt.data({'local_ips': [mock.sentinel.ip0, mock.sentinel.ip1], 'dest_ips': [mock.sentinel.ip2, mock.sentinel.ip3], 'expected_local': False}, {'local_ips': [mock.sentinel.ip0, mock.sentinel.ip1], 'dest_ips': [mock.sentinel.ip1, mock.sentinel.ip3], 'expected_local': True}, {'local_ips': [], 'dest_ips': ['127.0.0.1'], 'expected_local': True})
@ddt.unpack
@mock.patch('os_win._utils.get_ips')
@mock.patch('socket.gethostname')
def test_is_local_share(self, mock_gethostname, mock_get_ips, local_ips, dest_ips, expected_local):
    fake_share_server = 'fake_share_server'
    fake_share = '\\\\%s\\fake_share' % fake_share_server
    mock_get_ips.side_effect = (local_ips, ['127.0.0.1', '::1'], dest_ips)
    self._smbutils._loopback_share_map = {}
    is_local = self._smbutils.is_local_share(fake_share)
    self.assertEqual(expected_local, is_local)
    self._smbutils.is_local_share(fake_share)
    mock_gethostname.assert_called_once_with()
    mock_get_ips.assert_has_calls([mock.call(mock_gethostname.return_value), mock.call('localhost'), mock.call(fake_share_server)])