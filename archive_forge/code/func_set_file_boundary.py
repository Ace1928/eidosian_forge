import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def set_file_boundary(self):
    self._file.set_boundary(self._boundary_trimmed)