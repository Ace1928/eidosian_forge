from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import rbd
from os_brick.initiator import linuxrbd
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(priv_rootwrap, 'execute', return_value=None)
def test_connect_local_volume(self, mock_execute):
    rbd_connector = rbd.RBDConnector(None, do_local_attach=True)
    conn = {'name': 'pool/image', 'auth_username': 'fake_user', 'hosts': ['192.168.10.2'], 'ports': ['6789']}
    device_info = rbd_connector.connect_volume(conn)
    execute_call1 = mock.call('which', 'rbd')
    cmd = ['rbd', 'map', 'image', '--pool', 'pool', '--id', 'fake_user', '--mon_host', '192.168.10.2:6789']
    execute_call2 = mock.call(*cmd, root_helper=None, run_as_root=True)
    mock_execute.assert_has_calls([execute_call1, execute_call2])
    expected_info = {'path': '/dev/rbd/pool/image', 'type': 'block'}
    self.assertEqual(expected_info, device_info)