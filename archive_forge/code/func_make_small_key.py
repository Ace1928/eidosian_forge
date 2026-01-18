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
def make_small_key(self):
    small_src_key_as_string = os.urandom(SMALL_KEY_SIZE)
    small_src_key = self._MakeKey(data=small_src_key_as_string)
    return (small_src_key_as_string, small_src_key)