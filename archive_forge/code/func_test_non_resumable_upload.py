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
def test_non_resumable_upload(self):
    """
        Tests that non-resumable uploads work
        """
    small_src_file_as_string, small_src_file = self.make_small_file()
    small_src_file.seek(0, os.SEEK_END)
    dst_key = self._MakeKey(set_contents=False)
    try:
        dst_key.set_contents_from_file(small_src_file)
        self.fail('should fail as need to rewind the filepointer')
    except AttributeError:
        pass
    dst_key.set_contents_from_file(small_src_file, rewind=True)
    self.assertEqual(SMALL_KEY_SIZE, dst_key.size)
    self.assertEqual(small_src_file_as_string, dst_key.get_contents_as_string())