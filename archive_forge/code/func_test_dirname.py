import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def test_dirname(self):
    dirname = urlutils.dirname
    if sys.platform == 'win32':
        self.assertRaises(urlutils.InvalidURL, dirname, 'file:///path/to/foo')
        self.assertEqual('file:///C|/', dirname('file:///C|/foo'))
        self.assertEqual('file:///C|/', dirname('file:///C|/'))
    else:
        self.assertEqual('file:///', dirname('file:///foo'))
        self.assertEqual('file:///', dirname('file:///'))
    self.assertEqual('http://host/path/to', dirname('http://host/path/to/foo'))
    self.assertEqual('http://host/path/to', dirname('http://host/path/to/foo/'))
    self.assertEqual('http://host/path/to/foo', dirname('http://host/path/to/foo/', exclude_trailing_slash=False))
    self.assertEqual('http://host/', dirname('http://host/path'))
    self.assertEqual('http://host/', dirname('http://host/'))
    self.assertEqual('http://host', dirname('http://host'))
    self.assertEqual('http:///nohost', dirname('http:///nohost/path'))
    self.assertEqual('random+scheme://user:pass@ahost:port/', dirname('random+scheme://user:pass@ahost:port/path'))
    self.assertEqual('random+scheme://user:pass@ahost:port/', dirname('random+scheme://user:pass@ahost:port/path/'))
    self.assertEqual('random+scheme://user:pass@ahost:port/', dirname('random+scheme://user:pass@ahost:port/'))
    self.assertEqual('path/to', dirname('path/to/foo'))
    self.assertEqual('path/to', dirname('path/to/foo/'))
    self.assertEqual('path/to/foo', dirname('path/to/foo/', exclude_trailing_slash=False))
    self.assertEqual('path/..', dirname('path/../foo'))
    self.assertEqual('../path', dirname('../path/foo'))