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
def test_non_dry_mode_logs_translated_command_to_debug_logs(self):
    with _mock_boto_config({'GSUtil': {'use_gcloud_storage': 'always', 'hidden_shim_mode': 'no_fallback'}}):
        with util.SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            with mock.patch.object(self._fake_command, 'logger', autospec=True) as mock_logger:
                self._fake_command.translate_to_gcloud_storage_if_requested()
                mock_logger.debug.assert_has_calls([mock.call('Gcloud Storage Command: {} objects fake --zip opt1 -x arg1 arg2'.format(shim_util._get_gcloud_binary_path('fake_dir'))), mock.call('Environment variables for Gcloud Storage:'), mock.call('%s=%s', 'CLOUDSDK_METRICS_ENVIRONMENT', 'gsutil_shim'), mock.call('%s=%s', 'CLOUDSDK_STORAGE_RUN_BY_GSUTIL_SHIM', 'True')], any_order=True)