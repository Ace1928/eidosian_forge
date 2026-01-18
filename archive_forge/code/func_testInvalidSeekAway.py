from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import hashlib
import os
import pkgutil
from unittest import mock
from gslib.exception import CommandException
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.hashing_helper import HashingFileUploadWrapper
def testInvalidSeekAway(self):
    """Tests seeking to EOF and then reading without first doing a SEEK_SET."""
    tmp_file = self._GetTestFile()
    digesters = {'md5': GetMd5()}
    with open(tmp_file, 'rb') as stream:
        wrapper = HashingFileUploadWrapper(stream, digesters, {'md5': GetMd5}, self._dummy_url, self.logger)
        wrapper.read(TRANSFER_BUFFER_SIZE)
        wrapper.seek(0, os.SEEK_END)
        try:
            wrapper.read()
            self.fail('Expected CommandException for invalid seek.')
        except CommandException as e:
            self.assertIn('Read called on hashing file pointer in an unknown position', str(e))