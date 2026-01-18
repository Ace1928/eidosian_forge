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
def test_download_with_inconsistent_etag_in_tracker(self):
    """
        Tests resumable download with an inconsistent etag in tracker file
        """
    tmp_dir = self._MakeTempDir()
    dst_fp = self.make_dst_fp(tmp_dir)
    small_src_key_as_string, small_src_key = self.make_small_key()
    inconsistent_etag_tracker_file_name = os.path.join(tmp_dir, 'inconsistent_etag_tracker')
    f = open(inconsistent_etag_tracker_file_name, 'w')
    good_etag = small_src_key.etag.strip('"\'')
    new_val_as_list = []
    for c in reversed(good_etag):
        new_val_as_list.append(c)
    f.write('%s\n' % ''.join(new_val_as_list))
    f.close()
    res_download_handler = ResumableDownloadHandler(tracker_file_name=inconsistent_etag_tracker_file_name)
    small_src_key.get_contents_to_file(dst_fp, res_download_handler=res_download_handler)
    self.assertEqual(SMALL_KEY_SIZE, get_cur_file_size(dst_fp))
    self.assertEqual(small_src_key_as_string, small_src_key.get_contents_as_string())