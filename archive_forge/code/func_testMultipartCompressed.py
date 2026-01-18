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
def testMultipartCompressed(self):
    """Test that multipart uploads are compressed."""
    upload_config = base_api.ApiUploadInfo(accept=['*/*'], max_size=None, simple_multipart=True, simple_path=u'/upload')
    upload = transfer.Upload(stream=self.sample_stream, mime_type='text/plain', total_size=len(self.sample_data), close_stream=False, gzip_encoded=True)
    self.request.body = '{"body_field_one": 7}'
    upload.ConfigureRequest(upload_config, self.request, self.url_builder)
    self.assertEqual(self.url_builder.query_params['uploadType'], 'multipart')
    self.assertEqual(self.request.headers['Content-Encoding'], 'gzip')
    self.assertLess(len(self.request.body), len(self.sample_data))
    with gzip.GzipFile(fileobj=six.BytesIO(self.request.body)) as f:
        original = f.read()
        self.assertTrue(self.sample_data in original)