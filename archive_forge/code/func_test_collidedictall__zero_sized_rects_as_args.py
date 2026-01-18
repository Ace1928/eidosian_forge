import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedictall__zero_sized_rects_as_args(self):
    """Ensures collidedictall works correctly with zero sized rects
        as args.

        There should be no collisions with zero sized rects.
        """
    rect = Rect(0, 0, 20, 20)
    no_collide_item1 = ('no collide 1', Rect(2, 2, 0, 0))
    no_collide_item2 = ('no collide 2', Rect(2, 2, 2, 0))
    no_collide_item3 = ('no collide 3', Rect(2, 2, 0, 2))
    rect_values = dict((no_collide_item1, no_collide_item2, no_collide_item3))
    rect_keys = {tuple(v): k for k, v in rect_values.items()}
    expected_items = []
    for use_values in (True, False):
        d = rect_values if use_values else rect_keys
        collide_items = rect.collidedictall(d, use_values)
        self._assertCountEqual(collide_items, expected_items)