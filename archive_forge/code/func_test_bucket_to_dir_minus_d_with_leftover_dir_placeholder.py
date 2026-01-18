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
@unittest.skipUnless(UsingCrcmodExtension(), 'Test requires fast crcmod.')
@SkipForS3('The boto lib used for S3 does not handle objects starting with slashes if we use V4 signature.')
def test_bucket_to_dir_minus_d_with_leftover_dir_placeholder(self):
    """Tests that we correctly handle leftover dir placeholders.

    See comments in gslib.commands.rsync._FieldedListingIterator for details.
    """
    bucket_uri = self.CreateBucket()
    tmpdir = self.CreateTempDir()
    self.CreateObject(bucket_uri=bucket_uri, object_name='obj1', contents=b'obj1')
    key_uri = self.StorageUriCloneReplaceName(bucket_uri, '/')
    self.StorageUriSetContentsFromString(key_uri, '')

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check1():
        """Tests rsync works as expected."""
        self.RunGsUtil(['rsync', '-d', '-r', suri(bucket_uri), tmpdir], return_stderr=True)
        listing1 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        listing2 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        self.assertEqual(listing1, set(['/obj1', '//']))
        self.assertEqual(listing2, set(['/obj1']))
    _Check1()