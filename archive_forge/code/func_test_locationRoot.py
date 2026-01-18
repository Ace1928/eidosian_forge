import gc
from twisted.internet import defer
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import resource, util
from twisted.web.error import FlattenerError
from twisted.web.http import FOUND
from twisted.web.server import Request
from twisted.web.template import TagLoader, flattenString, tags
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from twisted.web.util import (
def test_locationRoot(self):
    """
        At the URL root issue a redirect to the current URL, removing any query
        string.
        """
    self.assertEqual(b'http://10.0.0.1/', self.doLocationTest(b'/'))
    self.assertEqual(b'http://10.0.0.1/', self.doLocationTest(b'/?biff=baff'))