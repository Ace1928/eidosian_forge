import unittest
from kivy.vector import Vector
from operator import truediv
def test_initializer_twoparameters(self):
    vector = Vector(1, 2)
    self.assertEqual(vector.x, 1)
    self.assertEqual(vector.y, 2)