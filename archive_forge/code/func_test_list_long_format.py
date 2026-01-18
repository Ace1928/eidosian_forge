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
@unittest.skipUnless(LIST_SERVICE_ACCOUNT and ALT_SERVICE_ACCOUNT, 'Test requires service account configuration.')
def test_list_long_format(self):
    self.setUpListTest()
    alt_access_id = self.CreateHelper(ALT_SERVICE_ACCOUNT)
    self.RunGsUtil(['hmac', 'update', '-s', 'INACTIVE', alt_access_id])
    stdout = self.RunGsUtil(['hmac', 'list', '-l'], return_stdout=True)
    try:
        self.assertIn(' ACTIVE', stdout)
        self.assertIn('INACTIVE', stdout)
        self.assertIn(ALT_SERVICE_ACCOUNT, stdout)
        self.assertIn(LIST_SERVICE_ACCOUNT, stdout)
        for key_metadata in self.ParseListOutput(stdout):
            self.AssertKeyMetadataMatches(key_metadata, state='.*')
    finally:
        self.CleanupHelper(alt_access_id)