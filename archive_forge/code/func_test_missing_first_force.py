from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from unittest import mock
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import MAX_BUCKET_LENGTH
from gslib.tests.testcase.integration_testcase import SkipForS3
import gslib.tests.util as util
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from gslib.utils.retry_util import Retry
def test_missing_first_force(self):
    bucket_uri = self.CreateBucket()
    object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='present', contents=b'foo')
    if self.multiregional_buckets:
        self.AssertNObjectsInBucket(bucket_uri, 1)
    if not self._use_gcloud_storage:
        self.RunGsUtil(['rm', '%s' % suri(bucket_uri, 'missing'), suri(object_uri)], expected_status=1)
    stderr = self.RunGsUtil(['rm', '-f', '%s' % suri(bucket_uri, 'missing'), suri(object_uri)], return_stderr=True, expected_status=1)
    self.assertEqual(stderr.count('Removing %s://' % self.default_provider), 1)
    self.RunGsUtil(['stat', suri(object_uri)], expected_status=1)