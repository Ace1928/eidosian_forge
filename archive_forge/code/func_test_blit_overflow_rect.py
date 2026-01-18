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
def test_blit_overflow_rect(self):
    """Full coverage w/ overflow, specified with a Rect"""
    result = self.dst_surface.blit(self.src_surface, pygame.Rect(-1, -1, 300, 300))
    self.assertIsInstance(result, pygame.Rect)
    self.assertEqual(result.size, (64, 64))
    for k in [(x, x) for x in range(64)]:
        self.assertEqual(self.dst_surface.get_at(k), (255, 255, 255))