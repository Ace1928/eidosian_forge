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
def build_input_file(self, size):
    buf = []
    for i in range(size):
        buf.append(str(random.randint(0, 9)))
    file_as_string = ''.join(buf)
    return (file_as_string, StringIO.StringIO(file_as_string))