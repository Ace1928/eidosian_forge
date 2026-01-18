import re
import sys
import traceback
from collections import OrderedDict
from textwrap import dedent
from types import FunctionType
from typing import Callable, Dict, List, NoReturn, Optional, Tuple, cast
from xml.etree.ElementTree import XML
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.defer import (
from twisted.python.failure import Failure
from twisted.test.testutils import XMLAssertionMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.web._flatten import BUFFER_SIZE
from twisted.web.error import FlattenerError, UnfilledSlot, UnsupportedType
from twisted.web.iweb import IRenderable, IRequest, ITemplateLoader
from twisted.web.template import (
from twisted.web.test._util import FlattenTestCase
def test_asynchronousFlattenError(self) -> None:
    """
        When flattening a renderer which raises an exception asynchronously,
        the error is reported when it occurs.
        """
    failing: Deferred[object] = Deferred()

    @implementer(IRenderable)
    class NotActuallyRenderable:
        """No methods provided; this will fail"""

        def __repr__(self) -> str:
            return '<unrenderable>'

        def lookupRenderMethod(self, name: str) -> Callable[[Optional[IRequest], Tag], Flattenable]:
            ...

        def render(self, request: Optional[IRequest]) -> Flattenable:
            return failing
    flattening = flattenString(None, [NotActuallyRenderable()])
    self.assertNoResult(flattening)
    exc = RuntimeError('example')
    failing.errback(exc)
    failure = self.failureResultOf(flattening, FlattenerError)
    self.assertRegex(str(failure.value), re.compile(dedent('                    Exception while flattening:\n                      \\[<unrenderable>\\]\n                      <unrenderable>\n                      <Deferred at .* current result: <twisted.python.failure.Failure builtins.RuntimeError: example>>\n                      File ".*", line \\d*, in _flattenTree\n                        element = await element.*\n                    '), flags=re.MULTILINE))
    self.assertIn('RuntimeError: example', str(failure.value))
    self.failureResultOf(failing, RuntimeError)