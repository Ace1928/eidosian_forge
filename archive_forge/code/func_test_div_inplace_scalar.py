import unittest
from kivy.vector import Vector
from operator import truediv
def test_div_inplace_scalar(self):
    finalVector = Vector(6, 6)
    finalVector /= 2
    self.assertEqual(finalVector.x, 3)
    self.assertEqual(finalVector.y, 3)