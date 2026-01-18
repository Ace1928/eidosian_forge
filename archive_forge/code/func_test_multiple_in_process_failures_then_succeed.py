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
def test_multiple_in_process_failures_then_succeed(self):
    """
        Tests resumable download that fails twice in one process, then completes
        """
    res_download_handler = ResumableDownloadHandler(num_retries=3)
    dst_fp = self.make_dst_fp()
    small_src_key_as_string, small_src_key = self.make_small_key()
    small_src_key.get_contents_to_file(dst_fp, res_download_handler=res_download_handler)
    self.assertEqual(SMALL_KEY_SIZE, get_cur_file_size(dst_fp))
    self.assertEqual(small_src_key_as_string, small_src_key.get_contents_as_string())