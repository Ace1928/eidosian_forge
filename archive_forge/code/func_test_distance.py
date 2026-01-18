import unittest
from kivy.vector import Vector
from operator import truediv
def test_distance(self):
    distance = Vector(10, 10).distance((5, 10))
    self.assertEqual(distance, 5)