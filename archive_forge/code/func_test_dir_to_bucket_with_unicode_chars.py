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
@unittest.skipIf(IS_WINDOWS, 'Windows Unicode support is problematic in Python 2.x.')
def test_dir_to_bucket_with_unicode_chars(self):
    """Tests that rsync -r works correctly with unicode filenames."""
    tmpdir = self.CreateTempDir()
    bucket_uri = self.CreateBucket()
    file_list = ['morales_suenÌƒos.jpg', 'morales_suenÌƒos.jpg', 'fooꝾoo']
    for filename in file_list:
        self.CreateTempFile(tmpdir=tmpdir, file_name=filename)
    expected_list_results = frozenset(['/morales_suenÌƒos.jpg', '/fooꝾoo']) if IS_OSX else frozenset(['/morales_suenÌƒos.jpg', '/morales_suenÌƒos.jpg', '/fooꝾoo'])

    @Retry(AssertionError, tries=3, timeout_secs=1)
    def _Check():
        """Tests rsync works as expected."""
        self.RunGsUtil(['rsync', '-r', tmpdir, suri(bucket_uri)])
        listing1 = TailSet(tmpdir, self.FlatListDir(tmpdir))
        listing2 = TailSet(suri(bucket_uri), self.FlatListBucket(bucket_uri))
        self.assertEqual(set(listing1), expected_list_results)
        self.assertEqual(set(listing2), expected_list_results)
    _Check()