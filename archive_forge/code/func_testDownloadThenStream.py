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
def testDownloadThenStream(self):
    bytes_http = object()
    http = object()
    download_stream = six.StringIO()
    download = transfer.Download.FromStream(download_stream, total_size=26)
    download.bytes_http = bytes_http
    base_url = 'https://part.one/'
    with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as make_request:
        make_request.return_value = http_wrapper.Response(info={'content-range': 'bytes 0-25/26', 'status': http_client.OK}, content=string.ascii_lowercase, request_url=base_url)
        request = http_wrapper.Request(url='https://part.one/')
        download.InitializeDownload(request, http=http)
        self.assertEqual(1, make_request.call_count)
        received_request = make_request.call_args[0][1]
        self.assertEqual(base_url, received_request.url)
        self.assertRangeAndContentRangeCompatible(received_request, make_request.return_value)
    with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as make_request:
        make_request.return_value = http_wrapper.Response(info={'status': http_client.REQUESTED_RANGE_NOT_SATISFIABLE}, content='error', request_url=base_url)
        download.StreamInChunks()
        self.assertEqual(1, make_request.call_count)
        received_request = make_request.call_args[0][1]
        self.assertEqual('bytes=26-', received_request.headers['range'])