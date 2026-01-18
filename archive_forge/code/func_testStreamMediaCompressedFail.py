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
def testStreamMediaCompressedFail(self):
    """Test that non-chunked uploads raise an exception.

        Ensure uploads with the compressed and resumable flags set called from
        StreamMedia raise an exception. Those uploads are unsupported.
        """
    upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, auto_transfer=True, gzip_encoded=True)
    upload.strategy = transfer.RESUMABLE_UPLOAD
    with mock.patch.object(http_wrapper, 'MakeRequest') as make_request:
        make_request.return_value = self.response
        upload.InitializeUpload(self.request, 'http')
        with self.assertRaises(exceptions.InvalidUserInputError):
            upload.StreamMedia()