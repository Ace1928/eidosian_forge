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
def testCopyingWildcardSpecifiedBucketSubDirToExistingDir(self):
    """Tests copying a wildcard-specified bucket subdir to a directory."""
    src_bucket_uri = self.CreateBucket(test_objects=['src_sub0dir/foo', 'src_sub1dir/foo', 'src_sub2dir/foo', 'src_sub3dir/foo'])
    dst_dir = self.CreateTempDir()
    for i, (final_src_char, final_dst_char) in enumerate((('', ''), ('', '/'), ('/', ''), ('/', '/'))):
        with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
            self.RunCommand('cp', ['-R', suri(src_bucket_uri, 'src_sub%d*' % i) + final_src_char, dst_dir + final_dst_char])
        actual = set((str(u) for u in self._test_wildcard_iterator(os.path.join(dst_dir, 'src_sub%ddir' % i, '**')).IterAll(expand_top_level_buckets=True)))
        expected = set([suri(dst_dir, 'src_sub%ddir' % i, 'foo')])
        self.assertEqual(expected, actual)