from __future__ import annotations
import re
import sys
import traceback
from twisted.python.compat import nativeString
from twisted.trial import unittest
from twisted.web import error
from twisted.web.template import Tag
def test_noMessageValidStatusNoLocation(self) -> None:
    """
        If no C{message} argument is passed to the L{InfiniteRedirection}
        constructor and C{location} is also empty and the C{code} argument is a
        valid HTTP status code, C{code} is mapped to a descriptive string to
        which C{message} is assigned without trying to include an empty
        location.
        """
    e = error.InfiniteRedirection(b'200')
    self.assertEqual(e.message, b'OK')