import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testConstructionVector2(self):
    v = Vector2(Vector2(1.2, 3.4))
    self.assertEqual(v.x, 1.2)
    self.assertEqual(v.y, 3.4)