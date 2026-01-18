import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_multipart_no_boundary(self):
    out = self.get_response(_multipart_no_boundary)
    out.read()
    self.assertRaises(errors.InvalidHttpResponse, out.seek, 1, 1)