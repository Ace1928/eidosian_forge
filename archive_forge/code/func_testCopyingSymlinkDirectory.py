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
@unittest.skipIf(system_util.IS_WINDOWS, 'os.symlink() is not available on Windows.')
def testCopyingSymlinkDirectory(self):
    """Tests that cp warns when copying a symlink directory."""
    bucket_uri = self.CreateBucket()
    tmpdir = self.CreateTempDir()
    tmpdir2 = self.CreateTempDir()
    subdir = os.path.join(tmpdir, 'subdir')
    os.mkdir(subdir)
    fpath1 = self.CreateTempFile(tmpdir=subdir, contents=b'foo')
    self.CreateTempFile(tmpdir=tmpdir2, contents=b'foo')
    os.mkdir(os.path.join(tmpdir, 'symlinkdir'))
    os.symlink(tmpdir2, os.path.join(subdir, 'symlinkdir'))
    mock_log_handler = self.RunCommand('cp', ['-r', tmpdir, suri(bucket_uri)], debug=1, return_log_handler=True)
    actual = set((str(u) for u in self._test_wildcard_iterator(suri(bucket_uri, '**')).IterAll(expand_top_level_buckets=True)))
    expected_object_path = suri(bucket_uri, os.path.basename(tmpdir), 'subdir', os.path.basename(fpath1))
    expected = set([expected_object_path])
    self.assertEqual(expected, actual)
    desired_msg = 'Skipping symlink directory "%s"' % os.path.join(subdir, 'symlinkdir')
    self.assertIn(desired_msg, mock_log_handler.messages['info'], '"%s" not found in mock_log_handler["info"]: %s' % (desired_msg, str(mock_log_handler.messages)))