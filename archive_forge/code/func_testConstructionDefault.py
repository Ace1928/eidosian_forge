import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testConstructionDefault(self):
    v = Vector3()
    self.assertEqual(v.x, 0.0)
    self.assertEqual(v.y, 0.0)
    self.assertEqual(v.z, 0.0)