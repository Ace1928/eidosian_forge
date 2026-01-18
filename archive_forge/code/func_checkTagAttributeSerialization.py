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
def checkTagAttributeSerialization(self, wrapTag: Callable[[Tag], Flattenable]) -> None:
    """
        Common implementation of L{test_serializedAttributeWithTag} and
        L{test_serializedAttributeWithDeferredTag}.

        @param wrapTag: A 1-argument callable that wraps around the attribute's
            value so other tests can customize it.
        @type wrapTag: callable taking L{Tag} and returning something
            flattenable
        """
    innerTag = tags.a('<>&"')
    outerTag = tags.img(src=wrapTag(innerTag))
    outer = self.assertFlattensImmediately(outerTag, b'<img src="&lt;a&gt;&amp;lt;&amp;gt;&amp;amp;&quot;&lt;/a&gt;" />')
    inner = self.assertFlattensImmediately(innerTag, b'<a>&lt;&gt;&amp;"</a>')
    self.assertXMLEqual(XML(outer).attrib['src'], inner)