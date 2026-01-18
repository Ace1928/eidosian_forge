import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_assert_access(self):

    def dumb_func(cursor_offset, line):
        return LinePart(0, 2, 'ab')
    self.func = dumb_func
    self.assertAccess('<a|b>d')