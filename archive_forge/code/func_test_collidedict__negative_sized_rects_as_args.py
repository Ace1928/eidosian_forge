import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedict__negative_sized_rects_as_args(self):
    """Ensures collidedict works correctly with negative sized rect args."""
    rect = Rect(0, 0, 10, 10)
    collide_item1 = ('collide 1', Rect(1, 1, -1, -1))
    no_collide_item1 = ('no collide 1', Rect(1, 1, -1, 0))
    no_collide_item2 = ('no collide 2', Rect(1, 1, 0, -1))
    rect_values = dict((collide_item1, no_collide_item1, no_collide_item2))
    rect_keys = {tuple(v): k for k, v in rect_values.items()}
    for use_values in (True, False):
        if use_values:
            expected_item = collide_item1
            d = rect_values
        else:
            expected_item = (tuple(collide_item1[1]), collide_item1[0])
            d = rect_keys
        collide_item = rect.collidedict(d, use_values)
        self.assertTupleEqual(collide_item, expected_item)