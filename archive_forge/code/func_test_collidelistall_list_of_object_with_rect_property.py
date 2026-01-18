import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidelistall_list_of_object_with_rect_property(self):
    r = Rect(1, 1, 10, 10)
    l = [self._ObjectWithRectProperty(Rect(1, 1, 10, 10)), self._ObjectWithRectProperty(Rect(5, 5, 10, 10)), self._ObjectWithRectProperty(Rect(15, 15, 1, 1)), self._ObjectWithRectProperty(Rect(2, 2, 1, 1))]
    self.assertEqual(r.collidelistall(l), [0, 1, 3])
    f = [self._ObjectWithRectProperty(Rect(50, 50, 1, 1)), self._ObjectWithRectProperty(Rect(20, 20, 5, 5))]
    self.assertFalse(r.collidelistall(f))