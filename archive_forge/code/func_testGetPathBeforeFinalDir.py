from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import gzip
import os
import six
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.exception import NO_URLS_MATCHED_GENERIC
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetDummyProjectForUnitTest
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils import copy_helper
from gslib.utils import system_util
def testGetPathBeforeFinalDir(self):
    """Tests GetPathBeforeFinalDir() (unit test)."""
    self.assertEqual('gs://', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/'), StorageUrlFromString('gs://bucket/obj')))
    self.assertEqual('gs://bucket', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/dir/'), StorageUrlFromString('gs://bucket/dir/obj')))
    self.assertEqual('gs://bucket', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/dir'), StorageUrlFromString('gs://bucket/dir/obj')))
    self.assertEqual('gs://bucket/dir', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/dir/obj'), StorageUrlFromString('gs://bucket/dir/obj')))
    self.assertEqual('gs://bucket/dir1', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/*/dir2'), StorageUrlFromString('gs://bucket/dir1/dir2/obj')))
    self.assertEqual('gs://bucket/dir1/dir2/dir3', copy_helper.GetPathBeforeFinalDir(StorageUrlFromString('gs://bucket/*/dir2/*/dir4'), StorageUrlFromString('gs://bucket/dir1/dir2/dir3/dir4/obj')))
    src_dir = self.CreateTempDir()
    subdir = os.path.join(src_dir, 'subdir')
    os.mkdir(subdir)
    self.assertEqual(suri(src_dir), copy_helper.GetPathBeforeFinalDir(StorageUrlFromString(suri(subdir)), StorageUrlFromString(suri(subdir, 'obj'))))