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
@unittest.skipIf(IS_WINDOWS, 'POSIX attributes not available on Windows.')
def test_bucket_to_dir_preserve_posix_errors(self):
    """Tests that rsync -P works properly with files that would be orphaned."""
    bucket_uri = self.CreateBucket()
    tmpdir = self.CreateTempDir()
    primary_gid = os.stat(tmpdir).st_gid
    non_primary_gid = util.GetNonPrimaryGid()
    subdir = os.path.join(tmpdir, 'subdir')
    os.mkdir(subdir)
    obj1 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1', mode='222', uid=os.getuid())
    obj2 = self.CreateObject(bucket_uri=bucket_uri, object_name='.obj2', contents=b'.obj2', gid=INVALID_GID(), mode='540')
    self.CreateObject(bucket_uri=bucket_uri, object_name='subdir/obj3', contents=b'subdir/obj3')
    obj6 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj6', contents=b'obj6', gid=INVALID_GID(), mode='440')
    obj7 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj7', contents=b'obj7', gid=non_primary_gid, mode='333')
    obj8 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj8', contents=b'obj8', uid=INVALID_UID())
    obj9 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj9', contents=b'obj9', uid=INVALID_UID(), mode='777')
    obj10 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj10', contents=b'obj10', gid=INVALID_GID(), uid=INVALID_UID())
    obj11 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj11', contents=b'obj11', gid=INVALID_GID(), uid=INVALID_UID(), mode='544')
    obj12 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj12', contents=b'obj12', uid=INVALID_UID(), gid=USER_ID)
    obj13 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj13', contents=b'obj13', uid=INVALID_UID(), gid=primary_gid, mode='644')
    obj14 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj14', contents=b'obj14', uid=USER_ID, gid=INVALID_GID())
    obj15 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj15', contents=b'obj15', uid=USER_ID, gid=INVALID_GID(), mode='655')
    obj16 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj16', contents=b'obj16', uid=USER_ID, mode='244')
    obj17 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj17', contents=b'obj17', uid=USER_ID, gid=primary_gid, mode='222')
    obj18 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj18', contents=b'obj18', uid=USER_ID, gid=non_primary_gid, mode='333')
    obj19 = self.CreateObject(bucket_uri=bucket_uri, object_name='obj19', contents=b'obj19', mode='222')
    self.CreateTempFile(tmpdir=tmpdir, file_name='.obj2', contents=b'.OBJ2')
    self.CreateTempFile(tmpdir=tmpdir, file_name='obj4', contents=b'obj4')
    self.CreateTempFile(tmpdir=subdir, file_name='obj5', contents=b'subdir/obj5')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Tests that an exception is thrown because files will be orphaned."""
        stderr = self.RunGsUtil(['rsync', '-P', '-r', suri(bucket_uri), tmpdir], expected_status=1, return_stderr=True)
        if self._use_gcloud_storage:
            gcloud_preserve_posix_warning = 'For preserving POSIX with rsync downloads, gsutil aborts if a single download will result in invalid destination POSIX. However, this'
            self.assertRegex(stderr, gcloud_preserve_posix_warning)
            read_regex = '{}#\\d+\\. User \\d+ owns file, but owner does not have read'
            gid_regex = "{}#\\d+ metadata doesn't exist on current system\\. GID"
            uid_regex = "{}#\\d+ metadata doesn't exist on current system\\. UID"
            self.assertRegex(stderr, read_regex.format('obj1'))
            self.assertRegex(stderr, gid_regex.format('obj2'))
            self.assertRegex(stderr, gid_regex.format('obj6'))
            self.assertRegex(stderr, read_regex.format('obj7'))
            self.assertRegex(stderr, uid_regex.format('obj8'))
            self.assertRegex(stderr, uid_regex.format('obj9'))
            self.assertRegex(stderr, uid_regex.format('obj10'))
            self.assertRegex(stderr, uid_regex.format('obj11'))
            self.assertRegex(stderr, uid_regex.format('obj12'))
            self.assertRegex(stderr, uid_regex.format('obj13'))
            self.assertRegex(stderr, gid_regex.format('obj14'))
            self.assertRegex(stderr, gid_regex.format('obj15'))
            self.assertRegex(stderr, read_regex.format('obj16'))
            self.assertRegex(stderr, read_regex.format('obj17'))
            self.assertRegex(stderr, read_regex.format('obj18'))
            self.assertRegex(stderr, read_regex.format('obj19'))
        else:
            self.assertIn(ORPHANED_FILE, stderr)
            self.assertRegex(stderr, BuildErrorRegex(obj1, POSIX_MODE_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj2, POSIX_GID_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj6, POSIX_GID_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj7, POSIX_MODE_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj8, POSIX_UID_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj9, POSIX_UID_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj10, POSIX_UID_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj11, POSIX_UID_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj12, POSIX_UID_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj13, POSIX_UID_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj14, POSIX_GID_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj15, POSIX_GID_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj16, POSIX_INSUFFICIENT_ACCESS_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj17, POSIX_MODE_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj18, POSIX_MODE_ERROR))
            self.assertRegex(stderr, BuildErrorRegex(obj19, POSIX_MODE_ERROR))
        listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11', '/obj12', '/obj13', '/obj14', '/obj15', '/obj16', '/obj17', '/obj18', '/obj19']))
        if self._use_gcloud_storage:
            self.assertEqual(listing2, set(['/.obj2', '/obj4', '/subdir/obj3', '/subdir/obj5']))
        else:
            self.assertEqual(listing2, set(['/.obj2', '/obj4', '/subdir/obj5']))
    _Check1()
    self._SetObjectCustomMetadataAttribute(self.default_provider, bucket_uri.bucket_name, '.obj2', MODE_ATTR, '640')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        """Tests that a file with a valid mode in metadata, nothing changed."""
        stderr = self.RunGsUtil(['rsync', '-P', '-r', suri(bucket_uri), tmpdir], expected_status=1, return_stderr=True)
        if self._use_gcloud_storage:
            self.assertIn("doesn't exist on current system. GID:", stderr)
        else:
            self.assertIn(ORPHANED_FILE, stderr)
        listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj6', '/obj7', '/obj8', '/obj9', '/obj10', '/obj11', '/obj12', '/obj13', '/obj14', '/obj15', '/obj16', '/obj17', '/obj18', '/obj19']))
        if self._use_gcloud_storage:
            self.assertEqual(listing2, set(['/.obj2', '/obj4', '/subdir/obj3', '/subdir/obj5']))
        else:
            self.assertEqual(listing2, set(['/.obj2', '/obj4', '/subdir/obj5']))
    _Check2()