import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedictall__invalid_dict_key_format(self):
    """Ensures collidedictall correctly handles dicts with invalid keys."""
    rect = Rect(0, 0, 10, 10)
    rect_values = {'collide': rect.copy()}
    with self.assertRaises(TypeError):
        collide_items = rect.collidedictall(rect_values)