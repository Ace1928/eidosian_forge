from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch('os_brick.initiator.connectors.rbd.RBDConnector._local_attach_volume')
def test_connect_volume_local(self, mock_local_attach):
    connector = rbd.RBDConnector(None, do_local_attach=True)
    res = connector.connect_volume(self.connection_properties)
    mock_local_attach.assert_called_once_with(self.connection_properties)
    self.assertIs(mock_local_attach.return_value, res)