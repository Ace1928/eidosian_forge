import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def testEquals(self):
    """check to see how the rect uses __eq__"""
    r1 = Rect(1, 2, 3, 4)
    r2 = Rect(10, 20, 30, 40)
    r3 = (10, 20, 30, 40)
    r4 = Rect(10, 20, 30, 40)

    class foo(Rect):

        def __eq__(self, other):
            return id(self) == id(other)

        def __ne__(self, other):
            return id(self) != id(other)

    class foo2(Rect):
        pass
    r5 = foo(10, 20, 30, 40)
    r6 = foo2(10, 20, 30, 40)
    self.assertNotEqual(r5, r2)
    self.assertEqual(r6, r2)
    rect_list = [r1, r2, r3, r4, r6]
    rect_list.remove(r2)
    rect_list.remove(r2)
    rect_list.remove(r2)
    rect_list.remove(r2)
    self.assertRaises(ValueError, rect_list.remove, r2)