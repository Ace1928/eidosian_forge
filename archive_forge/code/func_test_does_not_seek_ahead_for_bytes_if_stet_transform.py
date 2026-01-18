from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import stat
from gslib import storage_url
from gslib.tests import testcase
from gslib.tests import util
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils import system_util
from gslib.utils import temporary_file_util
def test_does_not_seek_ahead_for_bytes_if_stet_transform(self):
    """Tests that cp does not seek-ahead for bytes if file size will change."""
    tmpdir = self.CreateTempDir()
    for _ in range(3):
        self.CreateTempFile(tmpdir=tmpdir, contents=b'123456')
    bucket_uri = self.CreateBucket()
    with util.SetBotoConfigForTest([('GSUtil', 'task_estimation_threshold', '1'), ('GSUtil', 'task_estimation_force', 'True')]):
        stderr = self.RunGsUtil(['-m', '-o', 'GSUtil:stet_binary_path={}'.format(self.stet_binary_path), '-o', 'GSUtil:stet_config_path={}'.format(self.stet_config_path), 'cp', '-r', '--stet', tmpdir, suri(bucket_uri)], return_stderr=True)
        self.assertNotIn('18.0 B]', stderr)
        self.assertRegex(stderr, '2\\.\\d KiB]')