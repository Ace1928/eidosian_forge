from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_formatRootTagNoFilename(self) -> None:
    """
        The C{_formatRoot} method formats a C{Tag} with no filename information
        as 'Tag <tagName>'.
        """
    e = self.makeFlattenerError()
    self.assertEqual(e._formatRoot(Tag('a-tag')), 'Tag <a-tag>')