import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_invalid_response(self):
    self.assertRaises(errors.InvalidHttpResponse, self.get_response, _invalid_response)