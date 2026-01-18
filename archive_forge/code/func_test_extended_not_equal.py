import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_extended_not_equal(self):
    a = call(x=1).foo
    b = call(x=2).foo
    self.assertEqual(a, a)
    self.assertEqual(b, b)
    self.assertNotEqual(a, b)