import random
import unittest
from cirq_ft.linalg.lcu_util import (
def test_already_uniform(self):
    self.assertEqual(self.assertPreprocess(weights=[1]), ([0], [0]))
    self.assertEqual(self.assertPreprocess(weights=[1, 1]), ([0, 1], [0, 0]))
    self.assertEqual(self.assertPreprocess(weights=[1, 1, 1]), ([0, 1, 2], [0, 0, 0]))
    self.assertEqual(self.assertPreprocess(weights=[2, 2, 2]), ([0, 1, 2], [0, 0, 0]))