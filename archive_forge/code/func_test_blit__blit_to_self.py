import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def test_blit__blit_to_self(self):
    """Test that blit operation works on self, alpha value is
        correct, and that no RGB distortion occurs."""
    test_surface = pygame.Surface((128, 128), SRCALPHA, 32)
    area = test_surface.get_rect()
    for pt, test_color in test_utils.gradient(area.width, area.height):
        test_surface.set_at(pt, test_color)
    reference_surface = test_surface.copy()
    test_surface.blit(test_surface, (0, 0))
    for x in range(area.width):
        for y in range(area.height):
            r, g, b, a = reference_color = reference_surface.get_at((x, y))
            expected_color = (r, g, b, a + a * ((256 - a) // 256))
            self.assertEqual(reference_color, expected_color)
    self.assertEqual(reference_surface.get_rect(), test_surface.get_rect())