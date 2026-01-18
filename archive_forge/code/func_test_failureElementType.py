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
def test_failureElementType(self):
    """
        The I{type} renderer of L{FailureElement} renders the failure's
        exception type.
        """
    element = FailureElement(self.failure, TagLoader(tags.span(render='type')))
    d = flattenString(None, element)
    exc = b'builtins.Exception'
    d.addCallback(self.assertEqual, b'<span>' + exc + b'</span>')
    return d