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
def testStreamInChunks(self):
    """Test StreamInChunks."""
    resume_incomplete_responses = [http_wrapper.Response(info={'status': http_wrapper.RESUME_INCOMPLETE, 'location': 'http://www.uploads.com', 'range': '0-{}'.format(end)}, content='', request_url='http://www.uploads.com') for end in [199, 399, 599]]
    responses = [self.response] + resume_incomplete_responses + [self.response]
    bytes_http = httplib2.Http()
    upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, http=bytes_http)
    upload.strategy = transfer.RESUMABLE_UPLOAD
    upload.chunksize = 200
    with mock.patch.object(bytes_http, 'request') as make_request:
        make_request.side_effect = self.HttpRequestSideEffect(responses)
        upload.InitializeUpload(self.request, bytes_http)
        upload.StreamInChunks()
        self.assertEquals(make_request.call_count, len(responses))