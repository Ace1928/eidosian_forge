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
def testReadInChunksWithSeekToBeginning(self):
    """Reads one buffer, then seeks to 0 and reads chunks until the end."""
    tmp_file = self._GetTestFile()
    for initial_read in (TRANSFER_BUFFER_SIZE - 1, TRANSFER_BUFFER_SIZE, TRANSFER_BUFFER_SIZE + 1, TRANSFER_BUFFER_SIZE * 2 - 1, TRANSFER_BUFFER_SIZE * 2, TRANSFER_BUFFER_SIZE * 2 + 1, TRANSFER_BUFFER_SIZE * 3 - 1, TRANSFER_BUFFER_SIZE * 3, TRANSFER_BUFFER_SIZE * 3 + 1):
        for buffer_size in (TRANSFER_BUFFER_SIZE - 1, TRANSFER_BUFFER_SIZE, TRANSFER_BUFFER_SIZE + 1, self._temp_test_file_len - 1, self._temp_test_file_len, self._temp_test_file_len + 1):
            expect_exception = buffer_size < self._temp_test_file_len
            with open(tmp_file, 'rb') as stream:
                wrapper = ResumableStreamingJsonUploadWrapper(stream, buffer_size, test_small_buffer=True)
                wrapper.read(initial_read)
                try:
                    hex_digest = CalculateMd5FromContents(wrapper)
                    if expect_exception:
                        self.fail('Did not get expected CommandException for initial read size %s, buffer size %s' % (initial_read, buffer_size))
                except CommandException as e:
                    if not expect_exception:
                        self.fail('Got unexpected CommandException "%s" for initial read size %s, buffer size %s' % (str(e), initial_read, buffer_size))
            if not expect_exception:
                with open(tmp_file, 'rb') as stream:
                    actual = CalculateMd5FromContents(stream)
                self.assertEqual(actual, hex_digest, 'Digests not equal for initial read size %s, buffer size %s' % (initial_read, buffer_size))