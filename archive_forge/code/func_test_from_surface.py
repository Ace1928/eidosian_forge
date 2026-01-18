from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_from_surface(self):
    """Ensures from_surface creates a mask with the correct bits set.

        This test checks the masks created by the from_surface function using
        16 and 32 bit surfaces. Each alpha value (0-255) is tested against
        several different threshold values.
        Note: On 16 bit surface the requested alpha value can differ from what
              is actually set. This test uses the value read from the surface.
        """
    threshold_count = 256
    surface_color = [55, 155, 255, 0]
    expected_size = (11, 9)
    all_set_count = expected_size[0] * expected_size[1]
    none_set_count = 0
    for depth in (16, 32):
        surface = pygame.Surface(expected_size, SRCALPHA, depth)
        for alpha in range(threshold_count):
            surface_color[3] = alpha
            surface.fill(surface_color)
            if depth < 32:
                alpha = surface.get_at((0, 0))[3]
            threshold_test_values = {-1, 0, alpha - 1, alpha, alpha + 1, 255, 256}
            for threshold in threshold_test_values:
                msg = f'depth={depth}, alpha={alpha}, threshold={threshold}'
                if alpha > threshold:
                    expected_count = all_set_count
                else:
                    expected_count = none_set_count
                mask = pygame.mask.from_surface(surface=surface, threshold=threshold)
                self.assertIsInstance(mask, pygame.mask.Mask, msg)
                self.assertEqual(mask.get_size(), expected_size, msg)
                self.assertEqual(mask.count(), expected_count, msg)