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
def testLsBucketSubdirNonRecursive(self):
    """Test that ls of a bucket subdir returns expected results."""
    src_bucket_uri = self.CreateBucket(test_objects=['src_subdir/foo', 'src_subdir/nested/foo2'])
    output = self.RunCommand('ls', [suri(src_bucket_uri, 'src_subdir')], return_stdout=True)
    expected = set([suri(src_bucket_uri, 'src_subdir', 'foo'), suri(src_bucket_uri, 'src_subdir', 'nested') + src_bucket_uri.delim])
    expected.add('')
    actual = set([line.strip() for line in output.split('\n')])
    self.assertEqual(expected, actual)