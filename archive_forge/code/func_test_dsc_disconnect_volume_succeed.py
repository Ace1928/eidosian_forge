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
@mock.patch.object(priv_lightos, 'delete_dsc_file', return_value=True)
def test_dsc_disconnect_volume_succeed(self, mock_priv_lightos):
    self.connector.dsc_disconnect_volume(self._get_connection_info())