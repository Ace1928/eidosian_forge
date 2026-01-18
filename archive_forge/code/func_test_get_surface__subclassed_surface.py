import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_get_surface__subclassed_surface(self):
    """Ensure the surface attribute can handle subclassed surfaces."""
    expected_surface = SurfaceSubclass((5, 3), 0, 32)
    pixelarray = pygame.PixelArray(expected_surface)
    surface = pixelarray.surface
    self.assertIs(surface, expected_surface)
    self.assertIsInstance(surface, pygame.Surface)
    self.assertIsInstance(surface, SurfaceSubclass)