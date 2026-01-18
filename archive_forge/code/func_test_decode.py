import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_decode(self):
    self.assertEqual(decode('a<bd|c>d'), ((3, 'abdcd'), LinePart(1, 4, 'bdc')))
    self.assertEqual(decode('a|<bdc>d'), ((1, 'abdcd'), LinePart(1, 4, 'bdc')))
    self.assertEqual(decode('a<bdc>d|'), ((5, 'abdcd'), LinePart(1, 4, 'bdc')))