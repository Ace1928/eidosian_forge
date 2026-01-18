import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testConstructionKeywords(self):
    v = Vector3(x=1, y=2, z=3)
    self.assertEqual(v.x, 1.0)
    self.assertEqual(v.y, 2.0)
    self.assertEqual(v.z, 3.0)