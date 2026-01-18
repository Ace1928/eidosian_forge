import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_ipv6_port(self):
    parsed = urlutils.URL.from_string('http://[1:2:3::40]:80/one')
    self.assertEqual('http', parsed.scheme)
    self.assertEqual('1:2:3::40', parsed.host)
    self.assertIs(None, parsed.user)
    self.assertIs(None, parsed.password)
    self.assertEqual(80, parsed.port)
    self.assertEqual('/one', parsed.path)