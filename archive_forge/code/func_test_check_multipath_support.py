from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.windows import base as base_win_conn
from os_brick.tests.windows import fake_win_conn
from os_brick.tests.windows import test_base
@ddt.data({}, {'feature_available': True}, {'feature_available': False, 'enforce_multipath': True})
@ddt.unpack
@mock.patch.object(base_win_conn.utilsfactory, 'get_hostutils')
def test_check_multipath_support(self, mock_get_hostutils, feature_available=True, enforce_multipath=False):
    mock_hostutils = mock_get_hostutils.return_value
    mock_hostutils.check_server_feature.return_value = feature_available
    check_mpio = base_win_conn.BaseWindowsConnector.check_multipath_support
    if feature_available or not enforce_multipath:
        multipath_support = check_mpio(enforce_multipath=enforce_multipath)
        self.assertEqual(feature_available, multipath_support)
    else:
        self.assertRaises(exception.BrickException, check_mpio, enforce_multipath=enforce_multipath)
    mock_hostutils.check_server_feature.assert_called_once_with(mock_hostutils.FEATURE_MPIO)