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
@unittest.skipIf(IS_WINDOWS, 'POSIX attributes not available on Windows.')
@unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
def test_bucket_to_dir_preserve_posix_no_errors(self):
    """Tests that rsync -P works properly with default file attributes."""
    bucket_uri = self.CreateBucket()
    tmpdir = self.CreateTempDir()
    primary_gid = os.stat(tmpdir).st_gid
    non_primary_gid = util.GetNonPrimaryGid()
    subdir = os.path.join(tmpdir, 'subdir')
    os.mkdir(subdir)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1', mode='444')
    self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.obj2', gid=primary_gid)
    self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj3', contents=b'subdir/obj3', gid=non_primary_gid)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj6', contents=b'obj6', gid=primary_gid, mode='555')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj7', contents=b'obj7', gid=non_primary_gid, mode='444')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj8', contents=b'obj8', uid=USER_ID)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj9', contents=b'obj9', uid=USER_ID, mode='422')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj10', contents=b'obj10', uid=USER_ID, gid=primary_gid)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj11', contents=b'obj11', uid=USER_ID, gid=non_primary_gid)
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj12', contents=b'obj12', uid=USER_ID, gid=primary_gid, mode='400')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj13', contents=b'obj13', uid=USER_ID, gid=non_primary_gid, mode='533')
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj14', contents=b'obj14', uid=USER_ID, mode='444')
    self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.OBJ2')
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj4', contents=b'obj4')
    self.CreateTempFile(tmpdir=subdir, file_name='obj5', contents=b'subdir/obj5')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Verifies that all attributes were copied correctly when -P is used."""
        self.RunGsUtil(['rsync', '-P', '-r', suri(bucket_uri), tmpdir])
        listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11', '/obj12', '/obj13', '/obj14']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj4', '/subdir/obj5', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11', '/obj12', '/obj13', '/obj14']))
    _Check1()
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj1'), uid=os.getuid(), mode=292)
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, '.obj2'), gid=primary_gid, uid=os.getuid(), mode=DEFAULT_MODE)
    self.VerifyLocalPOSIXPermissions(os.path.join(subdir, 'obj3'), gid=non_primary_gid, mode=DEFAULT_MODE)
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj6'), gid=primary_gid, mode=365)
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj7'), gid=non_primary_gid, mode=292)
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj8'), gid=primary_gid, mode=DEFAULT_MODE)
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj9'), uid=USER_ID, mode=274)
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj10'), uid=USER_ID, gid=primary_gid, mode=DEFAULT_MODE)
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj11'), uid=USER_ID, gid=non_primary_gid, mode=DEFAULT_MODE)
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj12'), uid=USER_ID, gid=primary_gid, mode=256)
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj13'), uid=USER_ID, gid=non_primary_gid, mode=347)
    self.VerifyLocalPOSIXPermissions(os.path.join(tmpdir, 'obj14'), uid=USER_ID, mode=292)