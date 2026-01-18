import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_I(self):
    self.assertEqual(cursor('asd|fgh'), (3, 'asdfgh'))