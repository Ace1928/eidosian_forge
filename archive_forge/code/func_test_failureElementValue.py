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
def test_failureElementValue(self):
    """
        The I{value} renderer of L{FailureElement} renders the value's exception
        value.
        """
    element = FailureElement(self.failure, TagLoader(tags.span(render='value')))
    d = flattenString(None, element)
    d.addCallback(self.assertEqual, b'<span>This is a problem</span>')
    return d