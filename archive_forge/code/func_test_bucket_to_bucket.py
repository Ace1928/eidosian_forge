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
def test_bucket_to_bucket(self):
    """Tests that flat and recursive rsync between 2 buckets works correctly."""
    bucket1_uri = self.CreateBucket()
    bucket2_uri = self.CreateBucket()
    self.CreateObject(bucket_uri=bucket1_uri, object_name='obj1', contents=b'obj1')
    self.CreateObject(bucket_uri=bucket1_uri, object_name='.obj2', contents=b'.obj2', mtime=10)
    self.CreateObject(bucket_uri=bucket1_uri, object_name='subdir/obj3', contents=b'subdir/obj3')
    self.CreateObject(bucket_uri=bucket1_uri, object_name='obj6', contents=b'obj6_', mtime=100)
    self.CreateObject(bucket_uri=bucket2_uri, object_name='.obj2', contents=b'.OBJ2')
    self.CreateObject(bucket_uri=bucket2_uri, object_name='obj4', contents=b'obj4')
    self.CreateObject(bucket_uri=bucket2_uri, object_name='subdir/obj5', contents=b'subdir/obj5')
    self.CreateObject(bucket_uri=bucket2_uri, object_name='obj6', contents=b'obj6', mtime=100)

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Tests rsync works as expected."""
        self.RunGsUtil(['rsync', suri(bucket1_uri), suri(bucket2_uri)])
        listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
        listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
        self.assertEqual(listing1, set(['/obj1', '/.obj2', '/subdir/obj3', '/obj6']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/obj4', '/subdir/obj5', '/obj6']))
        self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket1_uri, '.obj2')], return_stdout=True))
        self.assertEqual('.obj2', self.RunGsUtil(['cat', suri(bucket2_uri, '.obj2')], return_stdout=True))
        self.assertEqual('obj6_', self.RunGsUtil(['cat', suri(bucket2_uri, 'obj6')], return_stdout=True))
        self._VerifyObjectMtime(bucket2_uri.bucket_name, '.obj2', '10')
    _Check1()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check2():
        stderr = self.RunGsUtil(['rsync', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
        self._VerifyNoChanges(stderr)
    _Check2()
    self.CreateObject(bucket_uri=bucket1_uri, object_name='obj6', contents=b'obj6')
    self.CreateObject(bucket_uri=bucket2_uri, object_name='obj7', contents=b'obj7')
    self.RunGsUtil(['rm', suri(bucket1_uri, 'obj1')])
    self.RunGsUtil(['rm', suri(bucket2_uri, '.obj2')])

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check3():
        self.RunGsUtil(['rsync', '-r', suri(bucket1_uri), suri(bucket2_uri)])
        listing1 = TailSet(suri(bucket1_uri), self.FlatListBucket(bucket1_uri))
        listing2 = TailSet(suri(bucket2_uri), self.FlatListBucket(bucket2_uri))
        self.assertEqual(listing1, set(['/.obj2', '/obj6', '/subdir/obj3']))
        self.assertEqual(listing2, set(['/obj1', '/.obj2', '/obj4', '/obj6', '/obj7', '/subdir/obj3', '/subdir/obj5']))
    _Check3()

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check4():
        stderr = self.RunGsUtil(['rsync', '-r', suri(bucket1_uri), suri(bucket2_uri)], return_stderr=True)
        self._VerifyNoChanges(stderr)
    _Check4()