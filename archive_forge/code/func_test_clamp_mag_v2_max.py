import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_clamp_mag_v2_max(self):
    v1 = Vector2(7, 2)
    v2 = v1.clamp_magnitude(5)
    v3 = v1.clamp_magnitude(0, 5)
    self.assertEqual(v2, v3)
    v1.clamp_magnitude_ip(5)
    self.assertEqual(v1, v2)
    v1.clamp_magnitude_ip(0, 5)
    self.assertEqual(v1, v2)
    expected_v2 = Vector2(4.807619738204116, 1.3736056394868903)
    self.assertEqual(expected_v2, v2)