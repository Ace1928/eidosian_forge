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
def testMovingObjectToBucketSubDir(self):
    """Tests moving an object to a bucket subdir."""
    src_bucket_uri = self.CreateBucket(test_objects=['obj0', 'obj1'])
    dst_bucket_uri = self.CreateBucket(test_objects=['dst_subdir0/existing_obj', 'dst_subdir1/existing_obj'])
    for i, final_dst_char in enumerate(('', '/')):
        self.RunCommand('mv', [suri(src_bucket_uri, 'obj%d' % i), suri(dst_bucket_uri, 'dst_subdir%d' % i) + final_dst_char])
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(dst_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    expected = set([suri(dst_bucket_uri, 'dst_subdir0', 'existing_obj'), suri(dst_bucket_uri, 'dst_subdir0', 'obj0'), suri(dst_bucket_uri, 'dst_subdir1', 'existing_obj'), suri(dst_bucket_uri, 'dst_subdir1', 'obj1')])
    self.assertEqual(expected, actual)
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(src_bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    self.assertEqual(actual, set())