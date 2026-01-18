from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_from_surface__with_colorkey_mask_pattern(self):
    """Ensures from_surface creates a mask with the correct bits set
        when the surface uses a colorkey.

        The surface is filled with alternating pixels of colorkey and
        non-colorkey colors, so the resulting masks are expected to have
        alternating bits set.
        """

    def alternate(func, set_value, unset_value, width, height):
        setbit = False
        for pos in ((x, y) for x in range(width) for y in range(height)):
            func(pos, set_value if setbit else unset_value)
            setbit = not setbit
    surface_color = (5, 10, 20)
    colorkey = (50, 60, 70)
    expected_size = (11, 2)
    expected_mask = pygame.mask.Mask(expected_size)
    alternate(expected_mask.set_at, 1, 0, *expected_size)
    expected_count = expected_mask.count()
    offset = (0, 0)
    for depth in (8, 16, 24, 32):
        msg = f'depth={depth}'
        surface = pygame.Surface(expected_size, 0, depth)
        alternate(surface.set_at, surface_color, colorkey, *expected_size)
        surface.set_colorkey(colorkey)
        mask = pygame.mask.from_surface(surface)
        self.assertIsInstance(mask, pygame.mask.Mask, msg)
        self.assertEqual(mask.get_size(), expected_size, msg)
        self.assertEqual(mask.count(), expected_count, msg)
        self.assertEqual(mask.overlap_area(expected_mask, offset), expected_count, msg)