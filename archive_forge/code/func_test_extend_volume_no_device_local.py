from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(rbd, 'open')
@mock.patch('os_brick.privileged.rbd.delete_if_exists')
@mock.patch.object(rbd.RBDConnector, '_find_root_device')
@mock.patch.object(rbd.RBDConnector, 'create_non_openstack_config')
def test_extend_volume_no_device_local(self, mock_config, mock_find, mock_delete, mock_open):
    mock_find.return_value = None
    connector = rbd.RBDConnector(None, do_local_attach=True)
    self.assertRaises(exception.BrickException, connector.extend_volume, self.connection_properties)
    mock_config.assert_called_once_with(self.connection_properties)
    mock_find.assert_called_once_with(self.connection_properties, mock_config.return_value)
    mock_delete.assert_called_once_with(mock_config.return_value)
    mock_open.assert_not_called()