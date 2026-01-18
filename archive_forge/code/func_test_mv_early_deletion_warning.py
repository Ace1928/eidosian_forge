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
@SkipForS3('Test is only relevant for gs storage classes.')
def test_mv_early_deletion_warning(self):
    """Tests that mv on a recent nearline object warns about early deletion."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('boto does not return object storage class')
    bucket_uri = self.CreateBucket(storage_class='NEARLINE')
    object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'obj')
    stderr = self.RunGsUtil(['mv', suri(object_uri), suri(bucket_uri, 'foo')], return_stderr=True)
    self.assertIn('Warning: moving nearline object %s may incur an early deletion charge, because the original object is less than 30 days old according to the local system time.' % suri(object_uri), stderr)