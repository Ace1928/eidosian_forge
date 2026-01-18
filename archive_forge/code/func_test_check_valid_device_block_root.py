from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(rbd.RBDConnector, '_check_valid_device')
@mock.patch('os_brick.privileged.rbd.check_valid_path')
@mock.patch.object(rbd, 'open')
def test_check_valid_device_block_root(self, mock_open, check_path, check_device):
    connector = rbd.RBDConnector(None)
    path = '/dev/rbd0'
    res = connector.check_valid_device(path, run_as_root=True)
    check_path.assert_called_once_with(path)
    self.assertEqual(check_path.return_value, res)
    mock_open.assert_not_called()
    check_device.assert_not_called()