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
def testNonRecursiveFileAndSameNameSubdir(self):
    """Tests copying a file and subdirectory of the same name without -R."""
    src_bucket_uri = self.CreateBucket(test_objects=['f1', 'f1/f2'])
    dst_dir = self.CreateTempDir()
    with SetBotoConfigForTest([('GSUtil', 'check_hashes', 'never')]):
        self.RunCommand('cp', [suri(src_bucket_uri, 'f1'), dst_dir])
    actual = list(self._test_wildcard_iterator('%s%s*' % (dst_dir, os.sep)).IterAll(expand_top_level_buckets=True))
    self.assertEqual(1, len(actual))
    self.assertEqual(suri(dst_dir, 'f1'), str(actual[0]))