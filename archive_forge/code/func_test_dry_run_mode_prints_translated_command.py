from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
from contextlib import contextmanager
import os
import re
import subprocess
from unittest import mock
from boto import config
from gslib import command
from gslib import command_argument
from gslib import exception
from gslib.commands import rsync
from gslib.commands import version
from gslib.commands import test
from gslib.cs_api_map import ApiSelector
from gslib.tests import testcase
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.tests import util
def test_dry_run_mode_prints_translated_command(self):
    """Should print the gcloud command and run gsutil."""
    with _mock_boto_config({'GSUtil': {'use_gcloud_storage': 'True', 'hidden_shim_mode': 'dry_run'}}):
        with util.SetEnvironmentForTest({'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            stdout, mock_log_handler = self.RunCommand('fake_shim', args=['arg1'], return_stdout=True, return_log_handler=True)
            self.assertIn('Gcloud Storage Command: {} objects fake arg1'.format(shim_util._get_gcloud_binary_path('fake_dir')), mock_log_handler.messages['info'])
            self.assertIn('FakeCommandWithGcloudStorageMap called'.format(shim_util._get_gcloud_binary_path('fake_dir')), stdout)