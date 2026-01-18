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
def test_dir_to_bucket_minus_u(self):
    """Tests that rsync -u works correctly."""
    tmpdir = self.CreateTempDir()
    dst_bucket = self.CreateBucket()
    ORIG_MTIME = 10
    self.CreateObject(bucket_uri=dst_bucket, object_name='obj1', contents=b'obj1-1', mtime=ORIG_MTIME)
    self.CreateObject(bucket_uri=dst_bucket, object_name='obj2', contents=b'obj2-1', mtime=ORIG_MTIME)
    self.CreateObject(bucket_uri=dst_bucket, object_name='obj3', contents=b'obj3-1', mtime=ORIG_MTIME)
    self.CreateObject(bucket_uri=dst_bucket, object_name='obj4', contents=b'obj4-1', mtime=ORIG_MTIME)
    self.CreateObject(bucket_uri=dst_bucket, object_name='obj5', contents=b'obj5-1', mtime=ORIG_MTIME)
    self.CreateObject(bucket_uri=dst_bucket, object_name='obj6', contents=b'obj6-1', mtime=ORIG_MTIME)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1-2', mtime=ORIG_MTIME - 1)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj2', contents=b'obj2-1', mtime=ORIG_MTIME - 1)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj3', contents=b'obj3-newer', mtime=ORIG_MTIME - 1)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj4', contents=b'obj4-2', mtime=ORIG_MTIME)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj5', contents=b'obj5-bigger', mtime=ORIG_MTIME)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj6', contents=b'obj6-1', mtime=ORIG_MTIME + 1)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check():
        self.RunGsUtil(['rsync', '-u', tmpdir, suri(dst_bucket)])
        self.assertEqual('obj1-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj1')], return_stdout=True))
        self.assertEqual('obj2-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj2')], return_stdout=True))
        self.assertEqual('obj3-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj3')], return_stdout=True))
        self.assertEqual('obj4-1', self.RunGsUtil(['cat', suri(dst_bucket, 'obj4')], return_stdout=True))
        self.assertEqual('obj5-bigger', self.RunGsUtil(['cat', suri(dst_bucket, 'obj5')], return_stdout=True))
        self._VerifyObjectMtime(dst_bucket.bucket_name, 'obj6', str(ORIG_MTIME + 1))
    _Check()