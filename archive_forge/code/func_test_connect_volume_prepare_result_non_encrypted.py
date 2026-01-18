import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@ddt.data(({}, {'type': 'block', 'path': '/dev/sda'}), ({'encrypted': False}, {'type': 'block', 'path': '/dev/sda'}), ({'encrypted': False}, {'type': 'block', 'path': b'/dev/sda'}), ({'encrypted': True}, {'type': 'block', 'path': io.StringIO()}))
@ddt.unpack
@mock.patch('os_brick.utils._symlink_name_from_device_path')
@mock.patch('os.path.realpath')
@mock.patch('os_brick.privileged.rootwrap.link_root')
def test_connect_volume_prepare_result_non_encrypted(self, conn_props, result, mock_link, mock_path, mock_get_symlink):
    """Test decorator for non encrypted devices or non host devices."""
    testing_self = mock.Mock()
    testing_self.connect_volume.return_value = result
    func = utils.connect_volume_prepare_result(testing_self.connect_volume)
    res = func(testing_self, conn_props)
    self.assertEqual(testing_self.connect_volume.return_value, res)
    testing_self.connect_volume.assert_called_once_with(testing_self, conn_props)
    mock_path.assert_not_called()
    mock_get_symlink.assert_not_called()
    mock_link.assert_not_called()