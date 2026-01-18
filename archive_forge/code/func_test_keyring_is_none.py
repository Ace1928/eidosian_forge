from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
def test_keyring_is_none(self):
    conn = rbd.RBDConnector(None)
    keyring = None
    keyring_data = '[client.cinder]\n  key = test\n'
    mockopen = mock.mock_open(read_data=keyring_data)
    mockopen.return_value.__exit__ = mock.Mock()
    with mock.patch('os_brick.initiator.connectors.rbd.open', mockopen, create=True):
        self.assertEqual(conn._check_or_get_keyring_contents(keyring, 'cluster', 'user'), keyring_data)
        self.assertEqual(conn._check_or_get_keyring_contents(keyring, 'cluster', None), '')