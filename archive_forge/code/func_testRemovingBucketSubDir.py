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
def testRemovingBucketSubDir(self):
    """Tests removing a bucket subdir."""
    dst_bucket_uri = self.CreateBucket(test_objects=['f0', 'dir0/f1', 'dir0/nested/f2', 'dir1/f1', 'dir1/nested/f2'])
    for i, final_src_char in enumerate(('', '/')):
        self.RunCommand('rm', ['-R', suri(dst_bucket_uri, 'dir%d' % i) + final_src_char])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(dst_bucket_uri, 'f0')])
    self.assertEqual(expected, actual)