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
def testLsBucketRecursive(self):
    """Test that ls -R of a bucket returns expected results."""
    src_bucket_uri = self.CreateBucket(test_objects=['foo1', 'd0/foo2', 'd1/d2/foo3'])
    output = self.RunCommand('ls', ['-R', suri(src_bucket_uri, '*')], return_stdout=True)
    expected = set([suri(src_bucket_uri, 'foo1'), suri(src_bucket_uri, 'd1', ':'), suri(src_bucket_uri, 'd1', 'd2', ':'), suri(src_bucket_uri, 'd1', 'd2', 'foo3'), suri(src_bucket_uri, 'd0', ':'), suri(src_bucket_uri, 'd0', 'foo2')])
    expected.add('')
    actual = set([line.strip() for line in output.split('\n')])
    self.assertEqual(expected, actual)