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