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
def test_lookup_with_slots(self):
    s = Slots()
    s.s1 = 's1'
    self.assertEqual(inspection.getattr_safe(s, 's1'), 's1')
    with self.assertRaises(AttributeError):
        inspection.getattr_safe(s, 's2')
    self.assertEqual(inspection.hasattr_safe(s, 's1'), True)
    self.assertEqual(inspection.hasattr_safe(s, 's2'), False)