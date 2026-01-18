from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
def test_keyring_raise_error(self):
    conn = rbd.RBDConnector(None)
    keyring = None
    mockopen = mock.mock_open()
    mockopen.return_value = ''
    with mock.patch('os_brick.initiator.connectors.rbd.open', mockopen, create=True) as mock_keyring_file:
        mock_keyring_file.side_effect = IOError
        self.assertRaises(exception.BrickException, conn._check_or_get_keyring_contents, keyring, 'cluster', 'user')