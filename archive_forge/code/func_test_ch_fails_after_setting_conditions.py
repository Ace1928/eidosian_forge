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
@unittest.skip('Disabled until all projects whitelisted for conditions.')
def test_ch_fails_after_setting_conditions(self):
    """Tests that if we "set" a policy with conditions, "ch" won't patch it."""
    print()
    self.RunGsUtil(['iam', 'set', '-e', '', self.new_bucket_policy_with_conditions_path, self.bucket.uri])
    stderr = self.RunGsUtil(['iam', 'ch', 'allUsers:objectViewer', self.bucket.uri], return_stderr=True, expected_status=1)
    self.assertIn('CommandException: Could not patch IAM policy for', stderr)
    self.assertIn('The resource had conditions present', stderr)
    stderr = self.RunGsUtil(['iam', 'ch', '-f', 'allUsers:objectViewer', self.bucket.uri], return_stderr=True, expected_status=1)
    self.assertIn('CommandException: Some IAM policies could not be patched', stderr)
    self.assertIn('Some resources had conditions', stderr)