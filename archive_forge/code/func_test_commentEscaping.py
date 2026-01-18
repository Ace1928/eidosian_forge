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
def test_commentEscaping(self) -> Deferred[List[bytes]]:
    """
        The data in a L{Comment} is escaped and mangled in the flattened output
        so that the result can be safely included in an HTML document.

        Test that C{>} is escaped when the sequence C{-->} is encountered
        within a comment, and that comments do not end with C{-}.
        """

    def verifyComment(c: bytes) -> None:
        self.assertTrue(c.startswith(b'<!--'), f'{c!r} does not start with the comment prefix')
        self.assertTrue(c.endswith(b'-->'), f'{c!r} does not end with the comment suffix')
        self.assertTrue(len(c) >= 7, f'{c!r} is too short to be a legal comment')
        content = c[4:-3]
        if b'foo' in content:
            self.assertIn(b'>', content)
        else:
            self.assertNotIn(b'>', content)
        if content:
            self.assertNotEqual(content[-1], b'-')
    results = []
    for c in ['', 'foo > bar', 'abracadabra-', 'not-->magic']:
        d = flattenString(None, Comment(c))
        d.addCallback(verifyComment)
        results.append(d)
    return gatherResults(results)