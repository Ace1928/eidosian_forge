import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def assertAccess(self, s):
    """Asserts that self.func matches as described
        by s, which uses a little language to describe matches:

        abcd<efg>hijklmnopqrstuvwx|yz
           /|\\ /|\\               /|\\
            |   |                 |
         the function should   the current cursor position
         match this "efg"      is between the x and y
        """
    (cursor_offset, line), match = decode(s)
    result = self.func(cursor_offset, line)
    self.assertEqual(result, match, "%s(%r) result\n%r (%r) doesn't match expected\n%r (%r)" % (self.func.__name__, line_with_cursor(cursor_offset, line), encode(cursor_offset, line, result), result, s, match))