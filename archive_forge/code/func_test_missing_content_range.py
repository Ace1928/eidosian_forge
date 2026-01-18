import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_missing_content_range(self):
    code, raw_headers, body = _single_range_no_content_range
    getheader = self._build_HTTPMessage(raw_headers)
    self.assertRaises(errors.InvalidHttpResponse, response.handle_response, 'http://bogus', code, getheader, BytesIO(body))