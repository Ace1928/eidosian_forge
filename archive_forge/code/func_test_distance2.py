import unittest
from kivy.vector import Vector
from operator import truediv
def test_distance2(self):
    distance = Vector(10, 10).distance2((5, 10))
    self.assertEqual(distance, 25)