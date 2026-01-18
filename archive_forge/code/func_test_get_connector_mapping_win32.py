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
@mock.patch('sys.platform', 'win32')
def test_get_connector_mapping_win32(self):
    mapping_win32 = connector.get_connector_mapping()
    self.assertIn('ISCSI', mapping_win32)
    self.assertIn('RBD', mapping_win32)
    self.assertNotIn('STORPOOL', mapping_win32)