import builtins
import json
import os
import subprocess
import sys
import tempfile
from unittest import mock
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.tests.unit import utils as test_utils
@mock.patch.object(os, 'access', autospec=True)
def test_make_configdrive_non_zero_returncode(self, mock_access, mock_popen):
    fake_process = mock.Mock(returncode=123)
    fake_process.communicate.return_value = ('', '')
    mock_popen.return_value = fake_process
    self.assertRaises(exc.CommandError, utils.make_configdrive, 'fake-dir')
    mock_access.assert_called_once_with('fake-dir', os.R_OK)
    mock_popen.assert_called_once_with(self.genisoimage_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    fake_process.communicate.assert_called_once_with()