from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch('oslo_utils.fileutils.delete_if_exists')
@mock.patch.object(rbd.RBDConnector, '_get_rbd_handle')
def test_extend_volume_handle_fail(self, mock_handle, mock_delete):
    mock_handle.return_value.seek.side_effect = ValueError
    connector = rbd.RBDConnector(None)
    self.assertRaises(ValueError, connector.extend_volume, self.connection_properties)
    mock_handle.assert_called_once_with(self.connection_properties)
    mock_handle.return_value.seek.assert_called_once_with(0, 2)
    mock_handle().tell.assert_not_called()
    mock_delete.assert_called_once_with(mock_handle.return_value.rbd_conf)
    mock_handle.return_value.close.assert_called_once_with()