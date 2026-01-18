import http.client as http_client
from io import BytesIO
from .. import errors, tests
from ..transport.http import response, urllib
from .file_utils import FakeReadFile
def test_can_read_at_first_access(self):
    """Test that the just created file can be read."""
    self.assertEqual(self.alpha, self._file.read())