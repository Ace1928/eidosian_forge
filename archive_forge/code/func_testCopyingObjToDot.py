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
def testCopyingObjToDot(self):
    """Tests that copying an object to . or ./ downloads to correct name."""
    src_bucket_uri = self.CreateBucket(test_objects=['f1'])
    dst_dir = self.CreateTempDir()
    for final_char in ('/', ''):
        with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
            self.RunCommand('cp', [suri(src_bucket_uri, 'f1'), '.%s' % final_char], cwd=dst_dir)
        actual = set()
        for dirname, dirnames, filenames in os.walk(dst_dir):
            for subdirname in dirnames:
                actual.add(os.path.join(dirname, subdirname))
            for filename in filenames:
                actual.add(os.path.join(dirname, filename))
        expected = set([os.path.join(dst_dir, 'f1')])
        self.assertEqual(expected, actual)