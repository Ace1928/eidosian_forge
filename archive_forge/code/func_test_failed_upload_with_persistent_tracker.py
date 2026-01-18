import errno
import random
import os
import time
from six import StringIO
import boto
from boto import storage_uri
from boto.gs.resumable_upload_handler import ResumableUploadHandler
from boto.exception import InvalidUriError
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from .cb_test_harness import CallbackTestHarness
from tests.integration.gs.testcase import GSTestCase
def test_failed_upload_with_persistent_tracker(self):
    """
        Tests that failed resumable upload leaves a correct tracker URI file
        """
    harness = CallbackTestHarness()
    tracker_file_name = self.make_tracker_file()
    res_upload_handler = ResumableUploadHandler(tracker_file_name=tracker_file_name, num_retries=0)
    small_src_file_as_string, small_src_file = self.make_small_file()
    small_src_file.seek(0)
    dst_key = self._MakeKey(set_contents=False)
    try:
        dst_key.set_contents_from_file(small_src_file, cb=harness.call, res_upload_handler=res_upload_handler)
        self.fail('Did not get expected ResumableUploadException')
    except ResumableUploadException as e:
        self.assertEqual(e.disposition, ResumableTransferDisposition.ABORT_CUR_PROCESS)
        self.assertTrue(os.path.exists(tracker_file_name))
        f = open(tracker_file_name)
        uri_from_file = f.readline().strip()
        f.close()
        self.assertEqual(uri_from_file, res_upload_handler.get_tracker_uri())