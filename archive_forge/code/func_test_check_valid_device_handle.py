from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@ddt.data(True, False)
@mock.patch.object(rbd.RBDConnector, '_check_valid_device')
@mock.patch('os_brick.privileged.rbd.check_valid_path')
@mock.patch.object(rbd, 'open')
def test_check_valid_device_handle(self, run_as_root, mock_open, check_path, check_device):
    connector = rbd.RBDConnector(None)
    res = connector.check_valid_device(mock.sentinel.handle, run_as_root=run_as_root)
    check_device.assert_called_once_with(mock.sentinel.handle)
    self.assertIs(check_device.return_value, res)
    mock_open.assert_not_called()
    check_path.assert_not_called()