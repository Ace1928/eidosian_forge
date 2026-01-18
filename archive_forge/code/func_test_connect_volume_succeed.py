import glob
import http.client
import queue
from unittest import mock
from unittest.mock import mock_open
from os_brick import exception
from os_brick.initiator.connectors import lightos
from os_brick.initiator import linuxscsi
from os_brick.privileged import lightos as priv_lightos
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(utils, 'get_host_nqn', return_value=FAKE_NQN)
@mock.patch.object(lightos.priv_lightos, 'move_dsc_file', return_value='/etc/discovery_client/discovery.d/v0')
@mock.patch.object(lightos.LightOSConnector, '_check_device_exists_using_dev_lnk', return_value='/dev/nvme0n1')
def test_connect_volume_succeed(self, mock_nqn, mock_move_file, mock_check_device):
    self.connector.connect_volume(self._get_connection_info())