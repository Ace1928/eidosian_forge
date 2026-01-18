import unittest
from kivy.vector import Vector
from operator import truediv
def test_angle(self):
    result = Vector(100, 0).angle((0, 100))
    self.assertAlmostEqual(result, -90.0)