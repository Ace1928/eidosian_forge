from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pkgutil
from six.moves import range
from gslib.exception import CommandException
from gslib.resumable_streaming_upload import ResumableStreamingJsonUploadWrapper
import gslib.tests.testcase as testcase
from gslib.utils.boto_util import GetJsonResumableChunkSize
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
from gslib.utils.hashing_helper import CalculateHashesFromContents
from gslib.utils.hashing_helper import CalculateMd5FromContents
from gslib.utils.hashing_helper import GetMd5
def testReadInChunks(self):
    tmp_file = self._GetTestFile()
    with open(tmp_file, 'rb') as stream:
        wrapper = ResumableStreamingJsonUploadWrapper(stream, TRANSFER_BUFFER_SIZE, test_small_buffer=True)
        hash_dict = {'md5': GetMd5()}
        CalculateHashesFromContents(wrapper, hash_dict)
    with open(tmp_file, 'rb') as stream:
        actual = CalculateMd5FromContents(stream)
    self.assertEqual(actual, hash_dict['md5'].hexdigest())