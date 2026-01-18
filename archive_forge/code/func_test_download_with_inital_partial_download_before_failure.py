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
def test_download_with_inital_partial_download_before_failure(self):
    """
        Tests resumable download that successfully downloads some content
        before it fails, then restarts and completes
        """
    harness = CallbackTestHarness(fail_after_n_bytes=LARGE_KEY_SIZE / 2)
    larger_src_key_as_string = os.urandom(LARGE_KEY_SIZE)
    larger_src_key = self._MakeKey(data=larger_src_key_as_string)
    res_download_handler = ResumableDownloadHandler(num_retries=1)
    dst_fp = self.make_dst_fp()
    larger_src_key.get_contents_to_file(dst_fp, cb=harness.call, res_download_handler=res_download_handler)
    self.assertEqual(LARGE_KEY_SIZE, get_cur_file_size(dst_fp))
    self.assertEqual(larger_src_key_as_string, larger_src_key.get_contents_as_string())
    self.assertTrue(len(harness.transferred_seq_before_first_failure) > 1 and len(harness.transferred_seq_after_first_failure) > 1)