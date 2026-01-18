import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedict__zero_sized_rects(self):
    """Ensures collidedict works correctly with zero sized rects.

        There should be no collisions with zero sized rects.
        """
    zero_rect1 = Rect(1, 1, 0, 0)
    zero_rect2 = Rect(1, 1, 1, 0)
    zero_rect3 = Rect(1, 1, 0, 1)
    zero_rect4 = Rect(1, 1, -1, 0)
    zero_rect5 = Rect(1, 1, 0, -1)
    no_collide_item1 = ('no collide 1', zero_rect1.copy())
    no_collide_item2 = ('no collide 2', zero_rect2.copy())
    no_collide_item3 = ('no collide 3', zero_rect3.copy())
    no_collide_item4 = ('no collide 4', zero_rect4.copy())
    no_collide_item5 = ('no collide 5', zero_rect5.copy())
    no_collide_item6 = ('no collide 6', Rect(0, 0, 10, 10))
    no_collide_item7 = ('no collide 7', Rect(0, 0, 2, 2))
    rect_values = dict((no_collide_item1, no_collide_item2, no_collide_item3, no_collide_item4, no_collide_item5, no_collide_item6, no_collide_item7))
    rect_keys = {tuple(v): k for k, v in rect_values.items()}
    for use_values in (True, False):
        d = rect_values if use_values else rect_keys
        for zero_rect in (zero_rect1, zero_rect2, zero_rect3, zero_rect4, zero_rect5):
            collide_item = zero_rect.collidedict(d, use_values)
            self.assertIsNone(collide_item)