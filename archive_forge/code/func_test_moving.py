from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
from gslib.cs_api_map import ApiSelector
from gslib.tests.test_cp import TestCpMvPOSIXBucketToLocalErrors
from gslib.tests.test_cp import TestCpMvPOSIXBucketToLocalNoErrors
from gslib.tests.test_cp import TestCpMvPOSIXLocalToBucketNoErrors
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
def test_moving(self):
    """Tests moving two buckets, one with 2 objects and one with 0 objects."""
    bucket1_uri = self.CreateBucket(test_objects=2)
    self.AssertNObjectsInBucket(bucket1_uri, 2)
    bucket2_uri = self.CreateBucket()
    self.AssertNObjectsInBucket(bucket2_uri, 0)
    objs = [self.StorageUriCloneReplaceKey(bucket1_uri, key).versionless_uri for key in bucket1_uri.list_bucket()]
    cmd = ['-m', 'mv'] + objs + [suri(bucket2_uri)]
    stderr = self.RunGsUtil(cmd, return_stderr=True)
    self.assertGreaterEqual(stderr.count('Copying'), 2, 'stderr did not contain 2 "Copying" lines:\n%s' % stderr)
    self.assertLessEqual(stderr.count('Copying'), 4, 'stderr did not contain <= 4 "Copying" lines:\n%s' % stderr)
    self.assertEqual(stderr.count('Copying') % 2, 0, 'stderr did not contain even number of "Copying" lines:\n%s' % stderr)
    self.assertEqual(stderr.count('Removing'), 2, 'stderr did not contain 2 "Removing" lines:\n%s' % stderr)
    self.AssertNObjectsInBucket(bucket1_uri, 0)
    self.AssertNObjectsInBucket(bucket2_uri, 2)
    objs = [self.StorageUriCloneReplaceKey(bucket2_uri, key).versionless_uri for key in bucket2_uri.list_bucket()]
    obj1 = objs[0]
    self.RunGsUtil(['rm', obj1])
    self.AssertNObjectsInBucket(bucket1_uri, 0)
    self.AssertNObjectsInBucket(bucket2_uri, 1)
    objs = [suri(self.StorageUriCloneReplaceKey(bucket2_uri, key)) for key in bucket2_uri.list_bucket()]
    cmd = ['-m', 'mv'] + objs + [suri(bucket1_uri)]
    stderr = self.RunGsUtil(cmd, return_stderr=True)
    self.assertGreaterEqual(stderr.count('Copying'), 1, 'stderr did not contain >= 1 "Copying" lines:\n%s' % stderr)
    self.assertLessEqual(stderr.count('Copying'), 2, 'stderr did not contain <= 2 "Copying" lines:\n%s' % stderr)
    self.assertEqual(stderr.count('Removing'), 1)
    self.AssertNObjectsInBucket(bucket1_uri, 1)
    self.AssertNObjectsInBucket(bucket2_uri, 0)