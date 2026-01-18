import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_is_normalized(self):
    self.assertEqual(self.v1.is_normalized(), False)
    v = self.v1.normalize()
    self.assertEqual(v.is_normalized(), True)
    self.assertEqual(self.e2.is_normalized(), True)
    self.assertEqual(self.zeroVec.is_normalized(), False)