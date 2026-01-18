import unittest
from kivy.vector import Vector
from operator import truediv
def test_initializer_oneparameter_as_int(self):
    with self.assertRaises(TypeError):
        Vector(1)