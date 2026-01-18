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
def testCopyingDirectoryToDirectory(self):
    """Tests copying from a directory to a directory."""
    src_dir = self.CreateTempDir(test_files=['foo', ('dir', 'foo2')])
    dst_dir = self.CreateTempDir()
    self.RunCommand('cp', ['-R', src_dir, dst_dir])
    actual = set((str(u) for u in self._test_wildcard_iterator('%s%s**' % (dst_dir, os.sep)).IterAll(expand_top_level_buckets=True)))
    src_dir_base = os.path.split(src_dir)[1]
    expected = set([suri(dst_dir, src_dir_base, 'foo'), suri(dst_dir, src_dir_base, 'dir', 'foo2')])
    self.assertEqual(expected, actual)