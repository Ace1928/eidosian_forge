import string
import unittest
import httplib2
import json
import mock
import six
from six.moves import http_client
from apitools.base.py import base_api
from apitools.base.py import exceptions
from apitools.base.py import gzip
from apitools.base.py import http_wrapper
from apitools.base.py import transfer
def testStreamInChunksCompressed(self):
    """Test that StreamInChunks will handle compression correctly."""
    upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, gzip_encoded=True)
    upload.strategy = transfer.RESUMABLE_UPLOAD
    upload.chunksize = len(self.sample_data)
    with mock.patch.object(transfer.Upload, '_Upload__SendMediaRequest') as mock_result, mock.patch.object(http_wrapper, 'MakeRequest') as make_request:
        mock_result.return_value = self.response
        make_request.return_value = self.response
        upload.InitializeUpload(self.request, 'http')
        upload.StreamInChunks()
        (request, _), _ = mock_result.call_args_list[0]
        self.assertTrue(mock_result.called)
        self.assertEqual(request.headers['Content-Encoding'], 'gzip')
        self.assertLess(len(request.body), len(self.sample_data))