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
def test_make_configdrive_oserror(self, mock_access, mock_popen):
    mock_access.return_value = True
    mock_popen.side_effect = OSError('boom')
    self.assertRaises(exc.CommandError, utils.make_configdrive, 'fake-dir')
    mock_access.assert_called_once_with('fake-dir', os.R_OK)
    mock_popen.assert_has_calls([mock.call(self.genisoimage_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE), mock.call(self.mkisofs_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE), mock.call(self.xorrisofs_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)])