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
def test_move_dirs_with_duplicate_file_names_to_bucket(self):
    """Tests moving multiple top-level items to a bucket."""
    bucket_uri = self.CreateBucket()
    dir_path = self.CreateTempDir(test_files=[('dir1', 'file.txt'), ('dir2', 'file.txt')])
    self.RunCommand('mv', [dir_path + '/*', suri(bucket_uri)])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(bucket_uri, 'dir1', 'file.txt'), suri(bucket_uri, 'dir2', 'file.txt')])
    self.assertEqual(actual, expected)