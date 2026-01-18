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
@mock.patch.object(priv_rootwrap, 'execute', side_effect=putils.ProcessExecutionError)
def test_brick_get_connector_properties_raise(self, mock_execute):
    self.assertRaises(putils.ProcessExecutionError, self._test_brick_get_connector_properties, True, True, None)