from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
from unittest import mock
import six
from gslib import command
from gslib.commands import rsync
from gslib.project_id import PopulateProjectId
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForGS
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import AuthorizeProjectToUseTestingKmsKey
from gslib.tests.util import TEST_ENCRYPTION_KEY_S3
from gslib.tests.util import TEST_ENCRYPTION_KEY_S3_MD5
from gslib.tests.util import BuildErrorRegex
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import ORPHANED_FILE
from gslib.tests.util import POSIX_GID_ERROR
from gslib.tests.util import POSIX_INSUFFICIENT_ACCESS_ERROR
from gslib.tests.util import POSIX_MODE_ERROR
from gslib.tests.util import POSIX_UID_ERROR
from gslib.tests.util import SequentialAndParallelTransfer
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TailSet
from gslib.tests.util import unittest
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.hashing_helper import SLOW_CRCMOD_RSYNC_WARNING
from gslib.utils.posix_util import ConvertDatetimeToPOSIX
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import NA_TIME
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.retry_util import Retry
from gslib.utils.system_util import IS_OSX
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils import shim_util
def test_invalid_src_mtime(self):
    """Tests that an exception is thrown if mtime cannot be cast as a long."""
    bucket1_uri = self.CreateBucket()
    bucket2_uri = self.CreateBucket()
    self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1', mtime='xyz')
    self.CreateObject(bucket_uri=bucket1_uri, object_name='obj2', contents=b'obj2', mtime=123)
    self.CreateObject(bucket_uri=bucket1_uri, object_name='obj3', contents=b'obj3', mtime=long(1234567891011))
    self.CreateObject(bucket_uri=bucket1_uri, object_name='obj4', contents=b'obj4', mtime=-100)
    self.CreateObject(bucket_uri=bucket1_uri, object_name='obj5', contents=b'obj5', mtime=-1)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        stderr = self.RunGsUtil(['rsync', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
        if self._use_gcloud_storage:
            self.assertRegex(stderr, 'obj1#\\d+ metadata did not contain a numeric value for goog-reserved-file-mtime')
            self.assertNotRegex(stderr, 'obj2#\\d+ metadata did not contain a numeric value for goog-reserved-file-mtime')
            self.assertRegex(stderr, 'obj3#\\d+ metadata that is more than one day in the future from the system time')
            self.assertRegex(stderr, 'Found negative time value in gs://.*/obj4')
            self.assertRegex(stderr, 'Found negative time value in gs://.*/obj5')
        else:
            self.assertIn('obj1 has an invalid mtime in its metadata', stderr)
            self.assertNotIn('obj2 has an invalid mtime in its metadata', stderr)
            self.assertIn('obj3 has an mtime more than 1 day from current system time', stderr)
            self.assertIn('obj4 has a negative mtime in its metadata', stderr)
            self.assertIn('obj5 has a negative mtime in its metadata', stderr)
    _Check1()