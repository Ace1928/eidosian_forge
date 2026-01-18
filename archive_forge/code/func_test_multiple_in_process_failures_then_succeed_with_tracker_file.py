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
def test_multiple_in_process_failures_then_succeed_with_tracker_file(self):
    """
        Tests resumable download that fails completely in one process,
        then when restarted completes, using a tracker file
        """
    harness = CallbackTestHarness(fail_after_n_bytes=LARGE_KEY_SIZE / 2, num_times_to_fail=2)
    larger_src_key_as_string = os.urandom(LARGE_KEY_SIZE)
    larger_src_key = self._MakeKey(data=larger_src_key_as_string)
    tmpdir = self._MakeTempDir()
    tracker_file_name = self.make_tracker_file(tmpdir)
    dst_fp = self.make_dst_fp(tmpdir)
    res_download_handler = ResumableDownloadHandler(tracker_file_name=tracker_file_name, num_retries=0)
    try:
        larger_src_key.get_contents_to_file(dst_fp, cb=harness.call, res_download_handler=res_download_handler)
        self.fail('Did not get expected ResumableDownloadException')
    except ResumableDownloadException as e:
        self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT_CUR_PROCESS)
        self.assertTrue(os.path.exists(tracker_file_name))
    larger_src_key.get_contents_to_file(dst_fp, cb=harness.call, res_download_handler=res_download_handler)
    self.assertEqual(LARGE_KEY_SIZE, get_cur_file_size(dst_fp))
    self.assertEqual(larger_src_key_as_string, larger_src_key.get_contents_as_string())
    self.assertFalse(os.path.exists(tracker_file_name))
    self.assertTrue(len(harness.transferred_seq_before_first_failure) > 1 and len(harness.transferred_seq_after_first_failure) > 1)