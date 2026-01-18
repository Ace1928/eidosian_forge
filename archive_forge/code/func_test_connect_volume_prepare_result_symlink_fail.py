import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@mock.patch('os_brick.utils._symlink_name_from_device_path')
@mock.patch('os.path.realpath')
@mock.patch('os_brick.privileged.rootwrap.link_root')
def test_connect_volume_prepare_result_symlink_fail(self, mock_link, mock_path, mock_get_symlink):
    """Test decorator for encrypted device failing on the symlink."""
    real_device = '/dev/md-6'
    connector_path = '/dev/md/alias'
    expected_symlink = '/dev/disk/by-id/os-brick_dev_md_alias'
    mock_path.return_value = real_device
    mock_get_symlink.return_value = expected_symlink
    testing_self = mock.Mock()
    connect_result = {'type': 'block', 'path': connector_path}
    mock_link.side_effect = ValueError
    testing_self.connect_volume.return_value = connect_result
    conn_props = {'encrypted': True}
    func = utils.connect_volume_prepare_result(testing_self.connect_volume)
    self.assertRaises(ValueError, func, testing_self, conn_props)
    testing_self.connect_volume.assert_called_once_with(testing_self, conn_props)
    mock_get_symlink.assert_called_once_with(connector_path)
    mock_link.assert_called_once_with(real_device, expected_symlink, force=True)
    testing_self.disconnect_volume.assert_called_once_with(connect_result, force=True, ignore_errors=True)