from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import math
import os
import pkgutil
import six
import gslib.cloud_api
from gslib.daisy_chain_wrapper import DaisyChainWrapper
from gslib.storage_url import StorageUrlFromString
import gslib.tests.testcase as testcase
from gslib.utils.constants import TRANSFER_BUFFER_SIZE
def testDownloadWithPartialWrite(self):
    """Tests unaligned writes to the download stream from GetObjectMedia."""
    with open(self.test_data_file, 'rb') as stream:
        chunk = stream.read(TRANSFER_BUFFER_SIZE)
    one_byte = chunk[0:1]
    chunk_minus_one_byte = chunk[1:TRANSFER_BUFFER_SIZE]
    half_chunk = chunk[0:TRANSFER_BUFFER_SIZE // 2]
    write_values_dict = {'First byte first chunk unaligned': (one_byte, chunk_minus_one_byte, chunk, chunk), 'Last byte first chunk unaligned': (chunk_minus_one_byte, chunk, chunk), 'First byte second chunk unaligned': (chunk, one_byte, chunk_minus_one_byte, chunk), 'Last byte second chunk unaligned': (chunk, chunk_minus_one_byte, one_byte, chunk), 'First byte final chunk unaligned': (chunk, chunk, one_byte, chunk_minus_one_byte), 'Last byte final chunk unaligned': (chunk, chunk, chunk_minus_one_byte, one_byte), 'Half chunks': (half_chunk, half_chunk, half_chunk), 'Many unaligned': (one_byte, half_chunk, one_byte, half_chunk, chunk, chunk_minus_one_byte, chunk, one_byte, half_chunk, one_byte)}
    upload_file = self.CreateTempFile()
    for case_name, write_values in six.iteritems(write_values_dict):
        expected_contents = b''
        for write_value in write_values:
            expected_contents += write_value
        mock_api = self.MockDownloadCloudApi(write_values)
        daisy_chain_wrapper = DaisyChainWrapper(self._dummy_url, len(expected_contents), mock_api, download_chunk_size=self.test_data_file_len)
        self._WriteFromWrapperToFile(daisy_chain_wrapper, upload_file)
        with open(upload_file, 'rb') as upload_stream:
            self.assertEqual(upload_stream.read(), expected_contents, 'Uploaded file contents for case %s did not match' % case_name)