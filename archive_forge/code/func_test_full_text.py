import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_full_text(self):
    out = self.get_response(_full_text_response)
    self.assertEqual(_full_text_response[2], out.read())