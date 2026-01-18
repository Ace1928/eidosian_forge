import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_full_text_no_content_type(self):
    code, raw_headers, body = _full_text_response_no_content_type
    getheader = self._build_HTTPMessage(raw_headers)
    out = response.handle_response('http://foo', code, getheader, BytesIO(body))
    self.assertEqual(body, out.read())