from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from gslib.cs_api_map import ApiSelector
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TEST_ENCRYPTION_KEY1
from gslib.tests.util import unittest
from gslib.utils import cat_helper
from gslib.utils import shim_util
from unittest import mock
@SkipForS3('S3 customer-supplied encryption keys are not supported.')
def test_cat_encrypted_object(self):
    if self.test_api == ApiSelector.XML:
        return unittest.skip('gsutil does not support encryption with the XML API')
    object_contents = b'0123456789'
    object_uri = self.CreateObject(object_name='foo', contents=object_contents, encryption_key=TEST_ENCRYPTION_KEY1)
    stderr = self.RunGsUtil(['cat', suri(object_uri)], expected_status=1, return_stderr=True)
    self.assertIn('No decryption key matches object', stderr)
    boto_config_for_test = [('GSUtil', 'encryption_key', TEST_ENCRYPTION_KEY1)]
    with SetBotoConfigForTest(boto_config_for_test):
        stdout = self.RunGsUtil(['cat', suri(object_uri)], return_stdout=True)
        self.assertEqual(stdout.encode('ascii'), object_contents)
        stdout = self.RunGsUtil(['cat', '-r', '1-3', suri(object_uri)], return_stdout=True)
        self.assertEqual(stdout, '123')