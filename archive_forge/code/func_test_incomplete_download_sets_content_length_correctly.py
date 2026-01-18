from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import email
from gslib.gcs_json_media import BytesTransferredContainer
from gslib.gcs_json_media import HttpWithDownloadStream
from gslib.gcs_json_media import UploadCallbackConnectionClassFactory
import gslib.tests.testcase as testcase
import httplib2
import io
import six
from six import add_move, MovedModule
from six.moves import http_client
from six.moves import mock
def test_incomplete_download_sets_content_length_correctly(self):
    expected_content_length = 100
    bytes_returned_by_server = 'byte count less than content length'
    http_response = mock.Mock(spec=http_client.HTTPResponse)
    http_response.reason = 'reason'
    http_response.version = 'version'
    http_response.status = http_client.OK
    http_response.read.side_effect = [bytes_returned_by_server, '']
    http_response.getheaders.return_value = [('Content-Length', expected_content_length)]
    http_response.getheader.return_value = expected_content_length
    headers_stream = io.BytesIO(b'Content-Length:%d' % expected_content_length)
    if six.PY2:
        http_response.msg = http_client.HTTPMessage(headers_stream)
    else:
        http_response.msg = http_client.parse_headers(headers_stream)
    mock_connection = mock.Mock(spec=http_client.HTTPConnection)
    mock_connection.getresponse.return_value = http_response
    http = HttpWithDownloadStream()
    http.stream = mock.Mock(spec=io.BufferedIOBase)
    http.stream.mode = 'wb'
    http._conn_request(mock_connection, 'uri', 'GET', 'body', 'headers')
    self.assertEqual(int(http_response.msg['content-length']), len(bytes_returned_by_server))