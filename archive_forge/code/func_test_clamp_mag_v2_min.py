import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_clamp_mag_v2_min(self):
    v1 = Vector2(1, 2)
    v2 = v1.clamp_magnitude(3, 5)
    v1.clamp_magnitude_ip(3, 5)
    expected_v2 = Vector2(1.3416407864998738, 2.6832815729997477)
    self.assertEqual(expected_v2, v2)
    self.assertEqual(expected_v2, v1)