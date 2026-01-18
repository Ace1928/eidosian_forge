import unittest
from traits.api import (
def test_arg_check_decorator(self):
    ac = ArgCheckDecorator(tc=self)
    for i in range(3):
        ac.value += 1
    self.assertEqual(ac.calls, 3 * 5)
    self.assertEqual(ac.value, 3)