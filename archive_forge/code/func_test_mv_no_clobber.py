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
def test_mv_no_clobber(self):
    """Tests mv with the -n option."""
    fpath1 = self.CreateTempFile(contents=b'data1')
    bucket_uri = self.CreateBucket()
    object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'data2')
    stderr = self.RunGsUtil(['mv', '-n', fpath1, suri(object_uri)], return_stderr=True)
    if self._use_gcloud_storage:
        self.assertIn('Skipping existing destination item (no-clobber): %s' % suri(object_uri), stderr)
    else:
        self.assertIn('Skipping existing item: %s' % suri(object_uri), stderr)
    self.assertNotIn('Removing %s' % suri(fpath1), stderr)
    contents = self.RunGsUtil(['cat', suri(object_uri)], return_stdout=True)
    self.assertEqual(contents, 'data2')