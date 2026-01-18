import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedict__no_collision(self):
    """Ensures collidedict returns None when no collisions."""
    rect = Rect(1, 1, 10, 10)
    no_collide_item1 = ('no collide 1', Rect(50, 50, 10, 10))
    no_collide_item2 = ('no collide 2', Rect(60, 60, 10, 10))
    no_collide_item3 = ('no collide 3', Rect(70, 70, 10, 10))
    rect_values = dict((no_collide_item1, no_collide_item2, no_collide_item3))
    rect_keys = {tuple(v): k for k, v in rect_values.items()}
    for use_values in (True, False):
        d = rect_values if use_values else rect_keys
        collide_item = rect.collidedict(d, use_values)
        self.assertIsNone(collide_item)