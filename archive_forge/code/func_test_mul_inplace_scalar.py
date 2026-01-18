import unittest
from kivy.vector import Vector
from operator import truediv
def test_mul_inplace_scalar(self):
    finalVector = Vector(2, 2)
    finalVector *= 3
    self.assertEqual(finalVector.x, 6)
    self.assertEqual(finalVector.y, 6)