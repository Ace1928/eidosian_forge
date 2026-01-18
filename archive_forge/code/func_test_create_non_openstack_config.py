from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch('os_brick.privileged.rbd.root_create_ceph_conf')
def test_create_non_openstack_config(self, mock_priv_create):
    res = rbd.RBDConnector.create_non_openstack_config(self.connection_properties)
    mock_priv_create.assert_called_once_with(self.hosts, self.ports, self.clustername, self.user, self.keyring)
    self.assertIs(mock_priv_create.return_value, res)