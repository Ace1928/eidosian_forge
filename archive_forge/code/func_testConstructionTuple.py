import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testConstructionTuple(self):
    v = Vector3((1.2, 3.4, 9.6))
    self.assertEqual(v.x, 1.2)
    self.assertEqual(v.y, 3.4)
    self.assertEqual(v.z, 9.6)