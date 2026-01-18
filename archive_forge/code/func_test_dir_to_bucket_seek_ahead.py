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
@SequentialAndParallelTransfer
@unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
def test_dir_to_bucket_seek_ahead(self):
    """Tests that rsync seek-ahead iterator works correctly."""

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Test estimating an rsync upload operation."""
        tmpdir = self.CreateTempDir()
        subdir = os.path.join(tmpdir, 'subdir')
        os.mkdir(subdir)
        self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1')
        self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2')
        self.CreateTempFile(tmpdir=subdir, file_name='obj3', contents=b'subdir/obj3')
        bucket_uri = self.CreateBucket()
        self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.OBJ2')
        self.CreateObject(bucket_uri=bucket_uri, object_name='obj4', contents=b'obj4')
        self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj5', contents=b'subdir/obj5')
        self.AssertNObjectsInBucket(bucket_uri, 3)
        with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
            stderr = self.RunGsUtil(['-m', 'rsync', '-d', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
            self.assertIn('Estimated work for this command: objects: 5, total size: 20', stderr)
            self.AssertNObjectsInBucket(bucket_uri, 3)
            stderr = self.RunGsUtil(['-m', 'rsync', '-d', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
            self.assertNotIn('Estimated work', stderr)
    _Check1()
    tmpdir = self.CreateTempDir(test_files=1)
    bucket_uri = self.CreateBucket()
    with SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '0'), ('GSUtil', 'task_estimation_force', 'True')]):
        stderr = self.RunGsUtil(['-m', 'rsync', '-d', '-r', tmpdir, suri(bucket_uri)], return_stderr=True)
        self.assertNotIn('Estimated work', stderr)