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
def test_blit__SRCALPHA_to_SRCALPHA_non_zero(self):
    """Tests blitting a nonzero alpha surface to another nonzero alpha surface
        both straight alpha compositing method. Test is fuzzy (+/- 1/256) to account for
        different implementations in SDL1 and SDL2.
        """
    size = (32, 32)

    def check_color_diff(color1, color2):
        """Returns True if two colors are within (1, 1, 1, 1) of each other."""
        for val in color1 - color2:
            if abs(val) > 1:
                return False
        return True

    def high_a_onto_low(high, low):
        """Tests straight alpha case. Source is low alpha, destination is high alpha"""
        high_alpha_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        low_alpha_surface = high_alpha_surface.copy()
        high_alpha_color = Color((high, high, low, high))
        low_alpha_color = Color((high, low, low, low))
        high_alpha_surface.fill(high_alpha_color)
        low_alpha_surface.fill(low_alpha_color)
        high_alpha_surface.blit(low_alpha_surface, (0, 0))
        expected_color = low_alpha_color + Color(tuple((x * (255 - low_alpha_color.a) // 255 for x in high_alpha_color)))
        self.assertTrue(check_color_diff(high_alpha_surface.get_at((0, 0)), expected_color))

    def low_a_onto_high(high, low):
        """Tests straight alpha case. Source is high alpha, destination is low alpha"""
        high_alpha_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        low_alpha_surface = high_alpha_surface.copy()
        high_alpha_color = Color((high, high, low, high))
        low_alpha_color = Color((high, low, low, low))
        high_alpha_surface.fill(high_alpha_color)
        low_alpha_surface.fill(low_alpha_color)
        low_alpha_surface.blit(high_alpha_surface, (0, 0))
        expected_color = high_alpha_color + Color(tuple((x * (255 - high_alpha_color.a) // 255 for x in low_alpha_color)))
        self.assertTrue(check_color_diff(low_alpha_surface.get_at((0, 0)), expected_color))
    for low_a in range(0, 128):
        for high_a in range(128, 256):
            high_a_onto_low(high_a, low_a)
            low_a_onto_high(high_a, low_a)