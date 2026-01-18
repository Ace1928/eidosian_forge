import unittest
from kivy.vector import Vector
from operator import truediv
def almost(a, b):
    self.assertIsNotNone(a)
    self.assertIsNotNone(b)
    self.assertAlmostEqual(a[0], b[0], places=0)
    self.assertAlmostEqual(a[1], b[1], places=0)