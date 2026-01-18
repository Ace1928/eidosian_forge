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
def test_check_valid_device_with_cmd_error(self):

    def raise_except(*args, **kwargs):
        raise putils.ProcessExecutionError
    self.connector = fake.FakeConnector(None)
    with mock.patch.object(self.connector, '_execute', side_effect=putils.ProcessExecutionError):
        self.assertFalse(self.connector.check_valid_device('/dev'))