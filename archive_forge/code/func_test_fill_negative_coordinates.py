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
def test_fill_negative_coordinates(self):
    color = (25, 25, 25, 25)
    color2 = (20, 20, 20, 25)
    fill_rect = pygame.Rect(-10, -10, 16, 16)
    s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
    r1 = s1.fill(color, fill_rect)
    c = s1.get_at((0, 0))
    self.assertEqual(c, color)
    s2 = s1.subsurface((5, 5, 5, 5))
    r2 = s2.fill(color2, (-3, -3, 5, 5))
    c2 = s1.get_at((4, 4))
    self.assertEqual(c, color)
    r3 = s2.fill(color2, (-30, -30, 5, 5))
    self.assertEqual(tuple(r3), (0, 0, 0, 0))