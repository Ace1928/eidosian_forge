from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_formatRootTagWithFilename(self) -> None:
    """
        The C{_formatRoot} method formats a C{Tag} with filename information
        using the filename, line, column, and tag information
        """
    e = self.makeFlattenerError()
    t = Tag('a-tag', filename='tpl.py', lineNumber=10, columnNumber=20)
    self.assertEqual(e._formatRoot(t), 'File "tpl.py", line 10, column 20, in "a-tag"')