import unittest
from traits.api import (
def arg_check1(self, new):
    self.calls += 1
    self.tc.assertEqual(new, self.exp_new)