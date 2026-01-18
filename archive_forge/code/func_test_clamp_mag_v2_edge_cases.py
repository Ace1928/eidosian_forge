import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_clamp_mag_v2_edge_cases(self):
    v1 = Vector2(1, 2)
    v2 = v1.clamp_magnitude(6, 6)
    v1.clamp_magnitude_ip(6, 6)
    self.assertEqual(v1, v2)
    self.assertAlmostEqual(v1.length(), 6)
    v2 = v1.clamp_magnitude(0)
    v1.clamp_magnitude_ip(0, 0)
    self.assertEqual(v1, v2)
    self.assertEqual(v1, Vector2())