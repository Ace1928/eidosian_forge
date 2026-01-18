from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
import json
import os
import subprocess
from gslib.commands import iam
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import shim_util
from gslib.utils.constants import UTF8
from gslib.utils.iam_helper import BindingsMessageToUpdateDict
from gslib.utils.iam_helper import BindingsDictToUpdateDict
from gslib.utils.iam_helper import BindingStringToTuple as bstt
from gslib.utils.iam_helper import DiffBindings
from gslib.utils.iam_helper import IsEqualBindings
from gslib.utils.iam_helper import PatchBindings
from gslib.utils.retry_util import Retry
from six import add_move, MovedModule
from six.moves import mock
@mock.patch.object(subprocess, 'run', autospec=True)
def test_iam_ch_continues_on_get_error(self, mock_run):
    original_policy = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allUsers']}]}
    new_policy = {'bindings': [{'role': 'roles/storage.modified-role', 'members': ['allAuthenticatedUsers', 'allUsers']}]}
    with SetBotoConfigForTest([('GSUtil', 'use_gcloud_storage', 'True')]):
        ls_process = subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps([{'url': 'gs://b/o1', 'type': 'cloud_object'}]))
        get_process = subprocess.CompletedProcess(args=[], returncode=1, stderr='An error.')
        ls_process2 = subprocess.CompletedProcess(args=[], returncode=1, stderr='Another error.')
        mock_run.side_effect = [ls_process, get_process, ls_process2]
        with SetEnvironmentForTest({'CLOUDSDK_CORE_PASS_CREDENTIALS_TO_GSUTIL': 'True', 'CLOUDSDK_ROOT_DIR': 'fake_dir'}):
            mock_log_handler = self.RunCommand('iam', ['ch', '-f', 'allAuthenticatedUsers:modified-role', 'gs://b/o1', 'gs://b/o2'], debug=1, return_log_handler=True)
        self.assertEqual(mock_run.call_args_list, [self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', 'gs://b/o1']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'objects', 'get-iam-policy', 'gs://b/o1', '--format=json']), self._get_run_call([shim_util._get_gcloud_binary_path('fake_dir'), 'storage', 'ls', '--json', 'gs://b/o2'])])
        error_lines = '\n'.join(mock_log_handler.messages['error'])
        self.assertIn('An error.', error_lines)
        self.assertIn('Another error.', error_lines)