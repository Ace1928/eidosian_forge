import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__kwarg_exceptions(self):
    """Ensure clipline handles incorrect keyword arguments"""
    r = Rect(2, 4, 6, 8)
    with self.assertRaises(TypeError):
        r.clipline(x1=0)
    with self.assertRaises(TypeError):
        r.clipline(first_coordinate=(1, 3, 5, 4), second_coordinate=(1, 2))
    with self.assertRaises(TypeError):
        r.clipline(first_coordinate=(1, 3), second_coordinate=(2, 2), x1=1)
    with self.assertRaises(TypeError):
        r.clipline(rect_arg=(1, 3, 5))
    with self.assertRaises(TypeError):
        r.clipline(rect_arg=(1, 3, 5, 4), second_coordinate=(2, 2))