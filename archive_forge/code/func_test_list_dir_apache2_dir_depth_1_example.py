import stat
from http.client import parse_headers
from io import StringIO
from breezy import errors, tests
from breezy.plugins.webdav import webdav
from breezy.tests import http_server
def test_list_dir_apache2_dir_depth_1_example(self):
    example = _get_list_dir_apache2_depth_1_allprop()
    self.assertEqual([('executable', False, 14, True), ('read-only', False, 42, False), ('titi', False, 6, False), ('toto', True, -1, False)], self._extract_dir_content_from_str(example))