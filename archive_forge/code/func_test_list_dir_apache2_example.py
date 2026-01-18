import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
def test_list_dir_apache2_example(self):
    example = _get_list_dir_apache2_depth_1_prop()
    self.assertRaises(errors.NotADirectory, self._extract_dir_content_from_str, example)