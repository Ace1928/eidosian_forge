import errno
import os
import re
import boto
from boto.s3.resumable_download_handler import get_cur_file_size
from boto.s3.resumable_download_handler import ResumableDownloadHandler
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableDownloadException
from .cb_test_harness import CallbackTestHarness
from tests.integration.gs.testcase import GSTestCase
def make_tracker_file(self, tmpdir=None):
    if not tmpdir:
        tmpdir = self._MakeTempDir()
    tracker_file = os.path.join(tmpdir, 'tracker')
    return tracker_file