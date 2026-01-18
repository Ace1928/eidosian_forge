import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_multi_squid_range(self):
    out = self.get_response(_multipart_squid_range_response)
    out.seek(0)
    out.read(100)
    out.seek(300)
    out.read(200)