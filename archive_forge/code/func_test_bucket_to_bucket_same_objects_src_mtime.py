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
def test_bucket_to_bucket_same_objects_src_mtime(self):
    """Tests bucket to bucket with mtime.

    Each has the same items but only the source has mtime stored in its
    metadata.
    Ensure that destination now also has the mtime of the files in its metadata.
    """
    src_bucket = self.CreateBucket()
    dst_bucket = self.CreateBucket()
    self.CreateObject(bucket_uri=src_bucket, object_name='obj1', contents=b'obj1', mtime=0)
    self.CreateObject(bucket_uri=src_bucket, object_name='subdir/obj2', contents=b'subdir/obj2', mtime=1)
    self.CreateObject(bucket_uri=dst_bucket, object_name='obj1', contents=b'obj1')
    self.CreateObject(bucket_uri=dst_bucket, object_name='subdir/obj2', contents=b'subdir/obj2')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Tests rsync works as expected."""
        self.RunGsUtil(['rsync', '-r', suri(src_bucket), suri(dst_bucket)])
        listing1 = TailSet(suri(src_bucket), self.FlatListBucket(src_bucket))
        self.assertEqual(listing1, set(['/obj1', '/subdir/obj2']))
    _Check1()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        stderr = self.RunGsUtil(['rsync', suri(src_bucket), suri(dst_bucket)], return_stderr=True)
        self._VerifyNoChanges(stderr)
    _Check2()
    if self._use_gcloud_storage:
        self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj1', NA_TIME, expected_present=False)
        self._VerifyObjectMtime(dst_bucket.bucket_name, 'subdir/obj2', NA_TIME, expected_present=False)
    else:
        self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj1', '0')
        self._VerifyObjectMtime(dst_bucket.bucket_name, 'subdir/obj2', '1')