import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidedictall__invalid_use_values_format(self):
    """Ensures collidedictall correctly handles invalid use_values
        parameters.
        """
    rect = Rect(0, 0, 1, 1)
    d = {}
    for invalid_param in (None, d, 1.1):
        with self.assertRaises(TypeError):
            collide_items = rect.collidedictall(d, invalid_param)