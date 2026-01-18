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
def test_dir_to_bucket_mtime(self):
    """Tests dir to bucket with mtime.

    Each has the same items, the source has mtime for all objects, whereas dst
    only has mtime for obj5 and obj6 to test for different a later mtime at src
    and the same mtime from src to dst, respectively. Ensure that destination
    now also has the mtime of the files in its metadata.
    """
    tmpdir = self.CreateTempDir()
    subdir = os.path.join(tmpdir, 'subdir')
    os.mkdir(subdir)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj1', contents=b'obj1', mtime=10)
    self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.obj2', mtime=10)
    self.CreateTempFile(tmpdir=subdir, file_name='obj3', contents=b'subdir/obj3', mtime=10)
    self.CreateTempFile(tmpdir=subdir, file_name='obj5', contents=b'subdir/obj5', mtime=15)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj6', contents=b'obj6', mtime=100)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj7', contents=b'obj7_', mtime=100)
    bucket_uri = self.CreateBucket()
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'OBJ1')
    self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.obj2')
    self._SetObjectCustomMetadataAttribute(self.default_provider, bucket_uri.bucket_name, '.obj2', 'test', 'test')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj4', contents=b'obj4')
    self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj5', contents=b'subdir/obj5', mtime=10)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj6', contents=b'OBJ6', mtime=100)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj7', contents=b'obj7', mtime=100)
    cumulative_stderr = set()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Tests rsync works as expected."""
        stderr = self.RunGsUtil(['rsync', '-r', '-d', tmpdir, suri(bucket_uri)], return_stderr=True)
        cumulative_stderr.update([s for s in stderr.splitlines() if s])
        listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/subdir/obj5', '/obj6', '/obj7']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj3', '/subdir/obj5', '/obj6', '/obj7']))
        self.assertEqual('OBJ6', self.RunGsUtil(['cat', suri(bucket_uri, 'obj6')], return_stdout=True))
        self.assertEqual('obj7_', self.RunGsUtil(['cat', suri(bucket_uri, 'obj7')], return_stdout=True))
    _Check1()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        stderr = self.RunGsUtil(['rsync', '-r', '-d', tmpdir, suri(bucket_uri)], return_stderr=True)
        self._VerifyNoChanges(stderr)
    _Check2()
    self._VerifyObjectMtime(bucket_uri.bucket_name, 'obj1', '10')
    self._VerifyObjectMtime(bucket_uri.bucket_name, '.obj2', '10')
    self._VerifyObjectMtime(bucket_uri.bucket_name, 'subdir/obj3', '10')
    self._VerifyObjectMtime(bucket_uri.bucket_name, 'subdir/obj5', '15')
    self._VerifyObjectMtime(bucket_uri.bucket_name, 'obj6', '100')
    copied_over_object_notice = "Copying whole file/object for %s instead of patching because you don't have owner permission on the object." % suri(bucket_uri, '.obj2')
    if copied_over_object_notice not in cumulative_stderr and (not self._use_gcloud_storage):
        self.VerifyObjectCustomAttribute(bucket_uri.bucket_name, '.obj2', 'test', 'test')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check4():
        """Tests rsync -c works as expected."""
        self.RunGsUtil(['rsync', '-r', '-d', '-c', tmpdir, suri(bucket_uri)])
        listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/subdir/obj5', '/obj6', '/obj7']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj3', '/subdir/obj5', '/obj6', '/obj7']))
        self.assertEqual('obj6', self.RunGsUtil(['cat', suri(bucket_uri, 'obj6')], return_stdout=True))
        self._VerifyObjectMtime(bucket_uri.bucket_name, 'obj6', '100')
    _Check4()