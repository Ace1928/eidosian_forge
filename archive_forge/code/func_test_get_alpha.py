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
def test_get_alpha(self):
    """Ensure a surface's alpha value can be retrieved."""
    s1 = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
    self.assertEqual(s1.get_alpha(), 255)
    for alpha in (0, 32, 127, 255):
        s1.set_alpha(alpha)
        for t in range(4):
            s1.set_alpha(s1.get_alpha())
        self.assertEqual(s1.get_alpha(), alpha)