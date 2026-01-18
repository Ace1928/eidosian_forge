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
def test_get_width__size_and_height(self):
    """Ensure a surface's size, width and height can be retrieved."""
    for w in range(0, 255, 32):
        for h in range(0, 127, 15):
            s = pygame.Surface((w, h))
            self.assertEqual(s.get_width(), w)
            self.assertEqual(s.get_height(), h)
            self.assertEqual(s.get_size(), (w, h))