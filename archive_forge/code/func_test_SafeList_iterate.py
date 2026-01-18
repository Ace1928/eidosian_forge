import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_SafeList_iterate(self):
    sl = SafeList(['1', 2, 3.0])
    self.assertTrue(isinstance(sl, list))
    it = sl.iterate()
    self.assertEqual(next(it), '1')
    self.assertEqual(next(it), 2)
    self.assertEqual(next(it), 3.0)