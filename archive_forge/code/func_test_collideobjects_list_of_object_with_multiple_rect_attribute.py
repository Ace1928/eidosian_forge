import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collideobjects_list_of_object_with_multiple_rect_attribute(self):
    r = Rect(1, 1, 10, 10)
    things = [self._ObjectWithMultipleRectAttribute(Rect(1, 1, 10, 10), Rect(5, 5, 1, 1), Rect(-73, 3, 3, 3)), self._ObjectWithMultipleRectAttribute(Rect(5, 5, 10, 10), Rect(-5, -5, 10, 10), Rect(3, 3, 3, 3)), self._ObjectWithMultipleRectAttribute(Rect(15, 15, 1, 1), Rect(100, 1, 1, 1), Rect(3, 83, 3, 3)), self._ObjectWithMultipleRectAttribute(Rect(2, 2, 1, 1), Rect(1, -81, 10, 10), Rect(3, 8, 3, 3))]
    self.assertEqual(r.collideobjects(things, key=lambda o: o.rect1), things[0])
    self.assertEqual(r.collideobjects(things, key=lambda o: o.rect2), things[0])
    self.assertEqual(r.collideobjects(things, key=lambda o: o.rect3), things[1])
    f = [self._ObjectWithMultipleRectAttribute(Rect(50, 50, 1, 1), Rect(11, 1, 1, 1), Rect(2, -32, 2, 2)), self._ObjectWithMultipleRectAttribute(Rect(20, 20, 5, 5), Rect(1, 11, 1, 1), Rect(-20, 2, 2, 2))]
    self.assertFalse(r.collideobjectsall(f, key=lambda o: o.rect1))
    self.assertFalse(r.collideobjectsall(f, key=lambda o: o.rect2))
    self.assertFalse(r.collideobjectsall(f, key=lambda o: o.rect3))