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
def test_get_source_file(self):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fodder')
    encoding = inspection.get_encoding_file(os.path.join(path, 'encoding_ascii.py'))
    self.assertEqual(encoding, 'ascii')
    encoding = inspection.get_encoding_file(os.path.join(path, 'encoding_latin1.py'))
    self.assertEqual(encoding, 'latin1')
    encoding = inspection.get_encoding_file(os.path.join(path, 'encoding_utf8.py'))
    self.assertEqual(encoding, 'utf-8')