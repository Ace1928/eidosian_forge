import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_compare__different_colors_within_distance(self):
    """Ensures compare works correctly with different colored surfaces
        and the color difference is within the given distance.
        """
    size = (3, 5)
    pixelarray_result_color = pygame.Color('white')
    surface_a_color = (127, 127, 127, 255)
    surface_b_color = (128, 127, 127, 255)
    for depth in (8, 16, 24, 32):
        expected_pixelarray_surface = pygame.Surface(size, depth=depth)
        expected_pixelarray_surface.fill(pixelarray_result_color)
        surf_a = expected_pixelarray_surface.copy()
        surf_a.fill(surface_a_color)
        expected_surface_a_color = surf_a.get_at((0, 0))
        pixelarray_a = pygame.PixelArray(surf_a)
        surf_b = expected_pixelarray_surface.copy()
        surf_b.fill(surface_b_color)
        expected_surface_b_color = surf_b.get_at((0, 0))
        pixelarray_b = pygame.PixelArray(surf_b)
        for distance in (0.2, 0.3, 0.5, 1.0):
            pixelarray_result = pixelarray_a.compare(pixelarray_b, distance=distance)
            self.assert_surfaces_equal(pixelarray_result.surface, expected_pixelarray_surface, (depth, distance))
            self.assert_surface_filled(pixelarray_a.surface, expected_surface_a_color, (depth, distance))
            self.assert_surface_filled(pixelarray_b.surface, expected_surface_b_color, (depth, distance))
        pixelarray_a.close()
        pixelarray_b.close()
        pixelarray_result.close()