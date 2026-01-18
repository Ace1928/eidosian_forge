import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_length_squared(self):
    self.assertEqual(Vector3(3, 4, 5).length_squared(), 3 * 3 + 4 * 4 + 5 * 5)
    self.assertEqual(Vector3(-3, 4, 5).length_squared(), -3 * -3 + 4 * 4 + 5 * 5)
    self.assertEqual(self.zeroVec.length_squared(), 0)