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
@mock.patch('os_brick.initiator.connector.platform.machine')
def test_get_connector_mapping(self, mock_platform_machine):
    mock_platform_machine.return_value = 'x86_64'
    mapping_x86 = connector.get_connector_mapping()
    mock_platform_machine.return_value = 'ppc64le'
    mapping_ppc = connector.get_connector_mapping()
    self.assertNotEqual(mapping_x86, mapping_ppc)
    mock_platform_machine.return_value = 's390x'
    mapping_s390 = connector.get_connector_mapping()
    self.assertNotEqual(mapping_x86, mapping_s390)
    self.assertNotEqual(mapping_ppc, mapping_s390)