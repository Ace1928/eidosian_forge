import unittest
from kivy.vector import Vector
from operator import truediv
def test_inbbox(self):
    bmin = (0, 0)
    bmax = (100, 100)
    result = Vector.in_bbox((50, 50), bmin, bmax)
    self.assertTrue(result)
    result = Vector.in_bbox((647, -10), bmin, bmax)
    self.assertFalse(result)