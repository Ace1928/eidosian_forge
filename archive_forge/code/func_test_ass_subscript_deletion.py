import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_ass_subscript_deletion(self):
    r = Rect(0, 0, 0, 0)
    with self.assertRaises(TypeError):
        del r[0]
    with self.assertRaises(TypeError):
        del r[0:2]
    with self.assertRaises(TypeError):
        del r[...]