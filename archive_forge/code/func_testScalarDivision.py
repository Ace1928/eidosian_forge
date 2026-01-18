import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testScalarDivision(self):
    v = self.v1 / self.s1
    self.assertTrue(isinstance(v, type(self.v1)))
    self.assertAlmostEqual(v.x, self.v1.x / self.s1)
    self.assertAlmostEqual(v.y, self.v1.y / self.s1)
    self.assertAlmostEqual(v.z, self.v1.z / self.s1)
    v = self.v1 // self.s2
    self.assertTrue(isinstance(v, type(self.v1)))
    self.assertEqual(v.x, self.v1.x // self.s2)
    self.assertEqual(v.y, self.v1.y // self.s2)
    self.assertEqual(v.z, self.v1.z // self.s2)