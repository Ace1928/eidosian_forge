import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testConstructionMissing(self):
    self.assertRaises(ValueError, Vector3, 1, 2)
    self.assertRaises(ValueError, Vector3, x=1, y=2)