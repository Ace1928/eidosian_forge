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
def test_bucket_to_dir_mtime(self):
    """Tests bucket to dir with mtime at the source."""
    bucket_uri = self.CreateBucket()
    tmpdir = self.CreateTempDir()
    subdir = os.path.join(tmpdir, 'subdir')
    os.mkdir(subdir)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1', mtime=5)
    self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.obj2', mtime=5)
    self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj3', contents=b'subdir/obj3')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj4', contents=b'OBJ4')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj6', contents=b'obj6', mtime=50)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj7', contents=b'obj7', mtime=5)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj8', contents=b'obj8', mtime=100)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj9', contents=b'obj9', mtime=25)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj10', contents=b'obj10')
    time_created = ConvertDatetimeToPOSIX(self._GetMetadataAttribute(bucket_uri.bucket_name, 'obj10', 'timeCreated'))
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj11', contents=b'obj11_', mtime=75)
    self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.OBJ2', mtime=10)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj4', contents=b'obj4', mtime=100)
    self.CreateTempFile(tmpdir=subdir, file_name='obj5', contents=b'subdir/obj5', mtime=10)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj6', contents=b'obj6', mtime=50)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj7', contents=b'OBJ7', mtime=50)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj8', contents=b'obj8', mtime=10)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj9', contents=b'OBJ9', mtime=25)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj10', contents=b'OBJ10', mtime=time_created)
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj11', contents=b'obj11', mtime=75)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Tests rsync works as expected."""
        self.RunGsUtil(['rsync', '-d', suri(bucket_uri), tmpdir])
        listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj4', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/obj4', '/subdir/obj5', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11']))
        with open(os.path.join(tmpdir, '.obj2')) as f:
            self.assertEqual('.obj2', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir, 'obj4')) as f:
            self.assertEqual('OBJ4', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir, 'obj9')) as f:
            self.assertEqual('OBJ9', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir, 'obj10')) as f:
            self.assertEqual('OBJ10', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir, 'obj11')) as f:
            self.assertEqual('obj11_', '\n'.join(f.readlines()))
    _Check1()

    def _Check2():
        """Verify mtime was set for objects at destination."""
        self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, 'obj1'))), 5)
        self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, '.obj2'))), 5)
        self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, 'obj6'))), 50)
        self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, 'obj8'))), 100)
        self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, 'obj9'))), 25)
    _Check2()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check3():
        """Tests rsync -c works as expected."""
        self.RunGsUtil(['rsync', '-r', '-d', '-c', suri(bucket_uri), tmpdir])
        listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj4', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj4', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11']))
        self.assertEqual('obj7', self.RunGsUtil(['cat', suri(bucket_uri, 'obj7')], return_stdout=True))
        self._VerifyObjectMtime(bucket_uri.bucket_name, 'obj7', '5')
        self.assertEqual(long(os.path.getmtime(os.path.join(tmpdir, 'obj7'))), 5)
        with open(os.path.join(tmpdir, 'obj9')) as f:
            self.assertEqual('obj9', '\n'.join(f.readlines()))
        with open(os.path.join(tmpdir, 'obj10')) as f:
            self.assertEqual('obj10', '\n'.join(f.readlines()))
    _Check3()