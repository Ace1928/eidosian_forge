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
def testMultipartEncoding(self):
    test_cases = ['line one\nFrom \nline two', u'name,main_ingredient\nRäksmörgås,Räkor\nBaguette,Bröd']
    for upload_contents in test_cases:
        multipart_body = '{"body_field_one": 7}'
        upload_bytes = upload_contents.encode('ascii', 'backslashreplace')
        upload_config = base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=True, resumable_path=u'/resumable/upload', simple_multipart=True, simple_path=u'/upload')
        url_builder = base_api._UrlBuilder('http://www.uploads.com')
        upload = transfer.Upload.FromStream(six.BytesIO(upload_bytes), 'text/plain', total_size=len(upload_bytes))
        http_request = http_wrapper.Request('http://www.uploads.com', headers={'content-type': 'text/plain'}, body=multipart_body)
        upload.ConfigureRequest(upload_config, http_request, url_builder)
        self.assertEqual('multipart', url_builder.query_params['uploadType'])
        rewritten_upload_contents = b'\n'.join(http_request.body.split(b'--')[2].splitlines()[1:])
        self.assertTrue(rewritten_upload_contents.endswith(upload_bytes))
        upload = transfer.Upload.FromStream(six.BytesIO(upload_bytes), 'text/plain', total_size=len(upload_bytes))
        http_request = http_wrapper.Request('http://www.uploads.com', headers={'content-type': 'text/plain'})
        upload.ConfigureRequest(upload_config, http_request, url_builder)
        self.assertEqual(url_builder.query_params['uploadType'], 'media')
        rewritten_upload_contents = http_request.body
        self.assertTrue(rewritten_upload_contents.endswith(upload_bytes))