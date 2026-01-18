import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedictall__invalid_dict_value_format(self):
    """Ensures collidedictall correctly handles dicts with invalid values."""
    rect = Rect(0, 0, 10, 10)
    rect_keys = {tuple(rect): 'collide'}
    with self.assertRaises(TypeError):
        collide_items = rect.collidedictall(rect_keys, 1)