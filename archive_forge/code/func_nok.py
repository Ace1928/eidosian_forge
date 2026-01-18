import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def nok(header_value):
    self.assertRaises(errors.InvalidHttpRange, f.set_range_from_header, header_value)