import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
def test_get_encoding_latin1(self):
    self.assertEqual(inspection.get_encoding(encoding_latin1), 'latin1')
    self.assertEqual(inspection.get_encoding(encoding_latin1.foo), 'latin1')