import platform
import sys
from unittest import mock
from oslo_concurrency import processutils as putils
from oslo_service import loopingcall
from os_brick import exception
from os_brick.initiator import connector
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fake
from os_brick.initiator.connectors import iscsi
from os_brick.initiator.connectors import nvmeof
from os_brick.initiator import linuxfc
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick import utils
def test_brick_get_connector_properties_connectors_called(self):
    """Make sure every connector is called."""
    mock_list = []
    for item in connector._get_connector_list():
        patched = mock.MagicMock()
        patched.platform = platform.machine()
        patched.os_type = sys.platform
        patched.__name__ = item
        patched.get_connector_properties.return_value = {}
        patcher = mock.patch(item, new=patched)
        patcher.start()
        self.addCleanup(patcher.stop)
        mock_list.append(patched)
    connector.get_connector_properties('sudo', MY_IP, True, True)
    for item in mock_list:
        assert item.get_connector_properties.called