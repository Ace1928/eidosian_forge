import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collideobjects_without_key(self):
    r = Rect(1, 1, 10, 10)
    types_to_test = [[Rect(50, 50, 1, 1), Rect(5, 5, 10, 10), Rect(4, 4, 1, 1)], [self._ObjectWithRectAttribute(Rect(50, 50, 1, 1)), self._ObjectWithRectAttribute(Rect(5, 5, 10, 10)), self._ObjectWithRectAttribute(Rect(4, 4, 1, 1))], [self._ObjectWithRectProperty(Rect(50, 50, 1, 1)), self._ObjectWithRectProperty(Rect(5, 5, 10, 10)), self._ObjectWithRectProperty(Rect(4, 4, 1, 1))], [self._ObjectWithCallableRectAttribute(Rect(50, 50, 1, 1)), self._ObjectWithCallableRectAttribute(Rect(5, 5, 10, 10)), self._ObjectWithCallableRectAttribute(Rect(4, 4, 1, 1))], [self._ObjectWithCallableRectAttribute(self._ObjectWithRectAttribute(Rect(50, 50, 1, 1))), self._ObjectWithCallableRectAttribute(self._ObjectWithRectAttribute(Rect(5, 5, 10, 10))), self._ObjectWithCallableRectAttribute(self._ObjectWithRectAttribute(Rect(4, 4, 1, 1)))], [(50, 50, 1, 1), (5, 5, 10, 10), (4, 4, 1, 1)], [((50, 50), (1, 1)), ((5, 5), (10, 10)), ((4, 4), (1, 1))], [[50, 50, 1, 1], [5, 5, 10, 10], [4, 4, 1, 1]], [Rect(50, 50, 1, 1), self._ObjectWithRectAttribute(Rect(5, 5, 10, 10)), (4, 4, 1, 1)]]
    for l in types_to_test:
        with self.subTest(type=l[0].__class__.__name__):
            actual = r.collideobjects(l)
            self.assertEqual(actual, l[1])
    types_to_test = [[Rect(50, 50, 1, 1), Rect(100, 100, 4, 4)], [self._ObjectWithRectAttribute(Rect(50, 50, 1, 1)), self._ObjectWithRectAttribute(Rect(100, 100, 4, 4))], [self._ObjectWithRectProperty(Rect(50, 50, 1, 1)), self._ObjectWithRectProperty(Rect(100, 100, 4, 4))], [self._ObjectWithCallableRectAttribute(Rect(50, 50, 1, 1)), self._ObjectWithCallableRectAttribute(Rect(100, 100, 4, 4))], [self._ObjectWithCallableRectAttribute(self._ObjectWithRectAttribute(Rect(50, 50, 1, 1))), self._ObjectWithCallableRectAttribute(self._ObjectWithRectAttribute(Rect(100, 100, 4, 4)))], [(50, 50, 1, 1), (100, 100, 4, 4)], [((50, 50), (1, 1)), ((100, 100), (4, 4))], [[50, 50, 1, 1], [100, 100, 4, 4]], [Rect(50, 50, 1, 1), [100, 100, 4, 4]]]
    for f in types_to_test:
        with self.subTest(type=f[0].__class__.__name__, expected=None):
            actual = r.collideobjects(f)
            self.assertEqual(actual, None)