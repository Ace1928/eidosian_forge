import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedictall__negative_sized_rects(self):
    """Ensures collidedictall works correctly with negative sized rects."""
    neg_rect = Rect(2, 2, -2, -2)
    collide_item1 = ('collide 1', neg_rect.copy())
    collide_item2 = ('collide 2', Rect(0, 0, 20, 20))
    no_collide_item1 = ('no collide 1', Rect(2, 2, 20, 20))
    rect_values = dict((collide_item1, collide_item2, no_collide_item1))
    value_collide_items = [collide_item1, collide_item2]
    rect_keys = {tuple(v): k for k, v in rect_values.items()}
    key_collide_items = [(tuple(v), k) for k, v in value_collide_items]
    for use_values in (True, False):
        if use_values:
            expected_items = value_collide_items
            d = rect_values
        else:
            expected_items = key_collide_items
            d = rect_keys
        collide_items = neg_rect.collidedictall(d, use_values)
        self._assertCountEqual(collide_items, expected_items)