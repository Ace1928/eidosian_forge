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
def test_blit_zero_overlap(self):
    """Testing zero-overlap condition."""
    result = self.dst_surface.blit(self.src_surface, dest=pygame.Rect((-256, -256, 1, 1)), area=pygame.Rect((2, 2, 256, 256)))
    self.assertIsInstance(result, pygame.Rect)
    self.assertEqual(result.size, (0, 0))
    for k in [(x, x) for x in range(64)]:
        self.assertEqual(self.dst_surface.get_at(k), (0, 0, 0))
    self.assertEqual(self.dst_surface.get_at((63, 0)), (0, 0, 0))
    self.assertEqual(self.dst_surface.get_at((0, 63)), (0, 0, 0))