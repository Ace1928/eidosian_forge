import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collideobjectsall_call_variants(self):
    r = Rect(1, 1, 10, 10)
    rects = [Rect(1, 2, 3, 4), Rect(10, 20, 30, 40)]
    objects = [self._ObjectWithMultipleRectAttribute(Rect(1, 2, 3, 4), Rect(10, 20, 30, 40), Rect(100, 200, 300, 400)), self._ObjectWithMultipleRectAttribute(Rect(1, 2, 3, 4), Rect(10, 20, 30, 40), Rect(100, 200, 300, 400))]
    r.collideobjectsall(rects)
    r.collideobjectsall(rects, key=None)
    r.collideobjectsall(objects, key=lambda o: o.rect1)
    self.assertRaises(TypeError, r.collideobjectsall, objects)