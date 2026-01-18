import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_angle_to(self):
    self.assertEqual(Vector3(1, 1, 0).angle_to((-1, 1, 0)), 90)
    self.assertEqual(Vector3(1, 0, 0).angle_to((0, 0, -1)), 90)
    self.assertEqual(Vector3(1, 0, 0).angle_to((-1, 0, 1)), 135)
    self.assertEqual(abs(Vector3(1, 0, 1).angle_to((-1, 0, -1))), 180)
    self.assertEqual(self.v1.rotate(self.v1.angle_to(self.v2), self.v1.cross(self.v2)).normalize(), self.v2.normalize())