import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@ddt.data({}, {'encrypted': False}, {'encrypted': True})
@mock.patch('os_brick.utils._symlink_name_from_device_path')
@mock.patch('os.path.realpath')
@mock.patch('os_brick.privileged.rootwrap.link_root')
def test_connect_volume_prepare_result_connect_fail(self, conn_props, mock_link, mock_path, mock_get_symlink):
    """Test decorator when decorated function fails."""
    testing_self = mock.Mock()
    testing_self.connect_volume.side_effect = ValueError
    func = utils.connect_volume_prepare_result(testing_self.connect_volume)
    self.assertRaises(ValueError, func, testing_self, conn_props)
    mock_link.assert_not_called()
    mock_path.assert_not_called()
    mock_get_symlink.assert_not_called()