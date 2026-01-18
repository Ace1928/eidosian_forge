from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import boto
import os
import re
from gslib.commands import hmac
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
def test_malformed_commands_that_cannot_be_translated_using_the_shim(self):
    if self._use_gcloud_storage:
        raise unittest.SkipTest('These commands cannot be translated using the shim.')
    params = [('hmac create -u email', 'requires a service account'), ('hmac update -s KENTUCKY', 'state flag value must be one of')]
    for command, gsutil_error_substr in params:
        stderr = self.RunGsUtil(command.split(), return_stderr=True, expected_status=1)
        self.assertIn(gsutil_error_substr, stderr)