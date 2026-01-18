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
def test_convert_alpha_init(self):
    """Ensure initialization exceptions are raised
        for surf.convert_alpha()."""
    pygame.display.quit()
    surf = pygame.Surface((1, 1))
    self.assertRaisesRegex(pygame.error, 'display initialized', surf.convert_alpha)
    pygame.display.init()
    try:
        self.assertRaisesRegex(pygame.error, 'No video mode', surf.convert_alpha)
        pygame.display.set_mode((640, 480))
        try:
            surf.convert_alpha()
        except pygame.error:
            self.fail('convert_alpha() should not raise an exception here.')
    finally:
        pygame.display.quit()