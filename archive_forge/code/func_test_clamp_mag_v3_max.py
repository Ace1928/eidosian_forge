import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_clamp_mag_v3_max(self):
    v1 = Vector3(7, 2, 2)
    v2 = v1.clamp_magnitude(5)
    v3 = v1.clamp_magnitude(0, 5)
    self.assertEqual(v2, v3)
    v1.clamp_magnitude_ip(5)
    self.assertEqual(v1, v2)
    v1.clamp_magnitude_ip(0, 5)
    self.assertEqual(v1, v2)
    expected_v2 = Vector3(4.635863249727653, 1.3245323570650438, 1.3245323570650438)
    self.assertEqual(expected_v2, v2)