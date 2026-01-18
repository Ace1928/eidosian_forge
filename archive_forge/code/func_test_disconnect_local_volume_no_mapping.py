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
def test_disconnect_local_volume_no_mapping(self, mock_execute):
    rbd_connector = rbd.RBDConnector(None, do_local_attach=True)
    conn = {'name': 'pool/not_mapped', 'auth_username': 'fake_user', 'hosts': ['192.168.10.2'], 'ports': ['6789']}
    mock_execute.return_value = ('\n{"0":{"pool":"pool","device":"/dev/rbd0","name":"pool-image"},\n "1":{"pool":"pool","device":"/dev/rdb1","name":"pool-image_2"}}', None)
    show_cmd = ['rbd', 'showmapped', '--format=json', '--id', 'fake_user', '--mon_host', '192.168.10.2:6789']
    rbd_connector.disconnect_volume(conn, None)
    mock_execute.assert_called_once_with(*show_cmd, root_helper=None, run_as_root=True)