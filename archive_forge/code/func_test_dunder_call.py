import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_dunder_call(self):
    m = MagicMock()
    m().foo()['bar']()
    self.assertEqual(m.mock_calls, [call(), call().foo(), call().foo().__getitem__('bar'), call().foo().__getitem__()()])
    m = MagicMock()
    m().foo()['bar'] = 1
    self.assertEqual(m.mock_calls, [call(), call().foo(), call().foo().__setitem__('bar', 1)])
    m = MagicMock()
    iter(m().foo())
    self.assertEqual(m.mock_calls, [call(), call().foo(), call().foo().__iter__()])