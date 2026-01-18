from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import subprocess
from unittest import mock
from gslib import exception
from gslib.tests import testcase
from gslib.utils import execution_util
@mock.patch.object(subprocess, 'Popen')
def testExternalCommandReturnsNoOutput(self, mock_Popen):
    mock_command_process = mock.Mock()
    mock_command_process.returncode = 1
    mock_command_process.communicate.return_value = (None, b'error')
    mock_Popen.return_value = mock_command_process
    with self.assertRaises(exception.ExternalBinaryError):
        execution_util.ExecuteExternalCommand(['fake-command'])
    mock_Popen.assert_called_once_with(['fake-command'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)