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
def testCopyingMultipleFilesToDirRecursive(self):
    """Tests copying multiple files with -R."""
    src_dir = self.CreateTempDir()
    src_file1 = self.CreateTempFile(tmpdir=src_dir, file_name='foo')
    src_file2 = self.CreateTempFile(tmpdir=src_dir, file_name='bar')
    dst_bucket_uri = self.CreateBucket()
    self.RunCommand('cp', ['-R', src_file1, src_file2, suri(dst_bucket_uri, 'dir/foo')])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(dst_bucket_uri, 'dir/foo/foo'), suri(dst_bucket_uri, 'dir/foo/bar')])
    self.assertEqual(expected, actual)