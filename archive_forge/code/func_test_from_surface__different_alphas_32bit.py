from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_from_surface__different_alphas_32bit(self):
    """Ensures from_surface creates a mask with the correct bits set
        when pixels have different alpha values (32 bits surfaces).

        This test checks the masks created by the from_surface function using
        a 32 bit surface. The surface is created with each pixel having a
        different alpha value (0-255). This surface is tested over a range
        of threshold values (0-255).
        """
    offset = (0, 0)
    threshold_count = 256
    surface_color = [10, 20, 30, 0]
    expected_size = (threshold_count, 1)
    expected_mask = pygame.Mask(expected_size, fill=True)
    surface = pygame.Surface(expected_size, SRCALPHA, 32)
    surface.lock()
    for a in range(threshold_count):
        surface_color[3] = a
        surface.set_at((a, 0), surface_color)
    surface.unlock()
    for threshold in range(threshold_count):
        msg = f'threshold={threshold}'
        expected_mask.set_at((threshold, 0), 0)
        expected_count = expected_mask.count()
        mask = pygame.mask.from_surface(surface, threshold)
        self.assertIsInstance(mask, pygame.mask.Mask, msg)
        self.assertEqual(mask.get_size(), expected_size, msg)
        self.assertEqual(mask.count(), expected_count, msg)
        self.assertEqual(mask.overlap_area(expected_mask, offset), expected_count, msg)