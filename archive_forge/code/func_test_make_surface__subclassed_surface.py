import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_make_surface__subclassed_surface(self):
    """Ensure make_surface can handle subclassed surfaces."""
    expected_size = (3, 5)
    expected_flags = 0
    expected_depth = 32
    original_surface = SurfaceSubclass(expected_size, expected_flags, expected_depth)
    pixelarray = pygame.PixelArray(original_surface)
    surface = pixelarray.make_surface()
    self.assertIsNot(surface, original_surface)
    self.assertIsInstance(surface, pygame.Surface)
    self.assertNotIsInstance(surface, SurfaceSubclass)
    self.assertEqual(surface.get_size(), expected_size)
    self.assertEqual(surface.get_flags(), expected_flags)
    self.assertEqual(surface.get_bitsize(), expected_depth)