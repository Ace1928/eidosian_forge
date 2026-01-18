from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_noMessageInvalidStatusLocationExists(self) -> None:
    """
        If no C{message} argument is passed to the L{InfiniteRedirection}
        constructor and C{code} isn't a valid HTTP status code, C{message} stays
        L{None}.
        """
    e = error.InfiniteRedirection(b'999', location=b'/foo')
    self.assertEqual(e.message, None)
    self.assertEqual(str(e), '999')