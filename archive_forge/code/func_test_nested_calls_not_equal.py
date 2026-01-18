import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_nested_calls_not_equal(self):
    a = call(x=1).foo().bar
    b = call(x=2).foo().bar
    self.assertEqual(a, a)
    self.assertEqual(b, b)
    self.assertNotEqual(a, b)