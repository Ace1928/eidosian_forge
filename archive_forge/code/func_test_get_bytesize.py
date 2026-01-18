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
def test_get_bytesize(self):
    """Ensure a surface's bit and byte sizes can be retrieved."""
    pygame.display.init()
    try:
        depth = 32
        depth_bytes = 4
        s1 = pygame.Surface((32, 32), pygame.SRCALPHA, depth)
        self.assertEqual(s1.get_bytesize(), depth_bytes)
        self.assertEqual(s1.get_bitsize(), depth)
        depth = 15
        depth_bytes = 2
        s1 = pygame.Surface((32, 32), 0, depth)
        self.assertEqual(s1.get_bytesize(), depth_bytes)
        self.assertEqual(s1.get_bitsize(), depth)
        depth = 12
        depth_bytes = 2
        s1 = pygame.Surface((32, 32), 0, depth)
        self.assertEqual(s1.get_bytesize(), depth_bytes)
        self.assertEqual(s1.get_bitsize(), depth)
        with self.assertRaises(pygame.error):
            surface = pygame.display.set_mode()
            pygame.display.quit()
            surface.get_bytesize()
    finally:
        pygame.display.quit()