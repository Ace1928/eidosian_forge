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
def test_blit_overflow_coord(self):
    """Full coverage w/ overflow, specified with Coordinate"""
    result = self.dst_surface.blit(self.src_surface, (0, 0))
    self.assertIsInstance(result, pygame.Rect)
    self.assertEqual(result.size, (64, 64))
    for k in [(x, x) for x in range(64)]:
        self.assertEqual(self.dst_surface.get_at(k), (255, 255, 255))