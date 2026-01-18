import random
import unittest
from cirq_ft.linalg.lcu_util import (
def test_donation(self):
    self.assertEqual(self.assertPreprocess(weights=[1, 2, 3]), ([2, 1, 2], [1, 0, 0]))
    self.assertEqual(self.assertPreprocess(weights=[3, 1, 2]), ([0, 0, 2], [0, 1, 0]))
    self.assertEqual(self.assertPreprocess(weights=[5, 1, 0]), ([0, 0, 0], [0, 1, 0]))