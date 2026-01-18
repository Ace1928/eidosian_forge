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
def test_failureElementTraceback(self):
    """
        The I{traceback} renderer of L{FailureElement} renders the failure's
        stack frames using L{_StackElement}.
        """
    element = FailureElement(self.failure)
    renderer = element.lookupRenderMethod('traceback')
    tag = tags.div()
    result = renderer(None, tag)
    self.assertIsInstance(result, _StackElement)
    self.assertIdentical(result.stackFrames, self.failure.frames)
    self.assertEqual([tag], result.loader.load())