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
def testRecursiveCopyFilesToExistingBucketSubDirInvalidSourceParent(self):
    """Tests recursive copy of invalid paths files to existing bucket subdir."""
    src_dir1 = self.CreateTempDir(test_files=['f1'])
    src_dir2 = os.path.join(src_dir1, 'nested')
    os.mkdir(src_dir2)
    self.CreateTempFile(tmpdir=src_dir2, file_name='f2')
    dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir/existing_obj'])
    for relative_path_string in ['.', '.' + os.sep, '..', '..' + os.sep]:
        with self.subTest(relative_path_string=relative_path_string):
            invalid_parent_dir = os.path.join(src_dir2, relative_path_string)
            with self.assertRaises(InvalidUrlError):
                self.RunCommand('cp', ['-R', src_dir1, invalid_parent_dir, suri(dst_bucket_uri, 'dst_subdir')])