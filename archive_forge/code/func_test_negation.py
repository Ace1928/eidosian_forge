import unittest
from kivy.vector import Vector
from operator import truediv
def test_negation(self):
    vector = -Vector(1, 1)
    self.assertEqual(vector.x, -1)
    self.assertEqual(vector.y, -1)