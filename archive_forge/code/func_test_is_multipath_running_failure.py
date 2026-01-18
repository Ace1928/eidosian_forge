import os
import os.path
import textwrap
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.initiator import linuxscsi
from os_brick.tests import base
@mock.patch.object(linuxscsi, 'LOG')
@mock.patch('os_brick.privileged.rootwrap.execute')
def test_is_multipath_running_failure(self, mock_exec, mock_log):
    mock_exec.side_effect = putils.ProcessExecutionError()
    self.assertRaises(putils.ProcessExecutionError, linuxscsi.LinuxSCSI.is_multipath_running, True, None, mock_exec)
    mock_log.error.assert_called_once()