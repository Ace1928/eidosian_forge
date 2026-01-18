import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testAttributeAccess(self):
    tmp = self.v1.x
    self.assertEqual(tmp, self.v1.x)
    self.assertEqual(tmp, self.v1[0])
    tmp = self.v1.y
    self.assertEqual(tmp, self.v1.y)
    self.assertEqual(tmp, self.v1[1])
    tmp = self.v1.z
    self.assertEqual(tmp, self.v1.z)
    self.assertEqual(tmp, self.v1[2])
    self.v1.x = 3.141
    self.assertEqual(self.v1.x, 3.141)
    self.v1.y = 3.141
    self.assertEqual(self.v1.y, 3.141)
    self.v1.z = 3.141
    self.assertEqual(self.v1.z, 3.141)

    def assign_nonfloat():
        v = Vector2()
        v.x = 'spam'
    self.assertRaises(TypeError, assign_nonfloat)