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
def test_make_configdrive(self, mock_popen):
    fake_process = mock.Mock(returncode=0)
    fake_process.communicate.return_value = ('', '')
    mock_popen.return_value = fake_process
    with utils.tempdir() as dirname:
        utils.make_configdrive(dirname)
    mock_popen.assert_called_once_with(self.genisoimage_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    fake_process.communicate.assert_called_once_with()