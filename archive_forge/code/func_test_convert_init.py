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
def test_convert_init(self):
    """Ensure initialization exceptions are raised
        for surf.convert()."""
    pygame.display.quit()
    surf = pygame.Surface((1, 1))
    self.assertRaisesRegex(pygame.error, 'display initialized', surf.convert)
    pygame.display.init()
    try:
        if os.environ.get('SDL_VIDEODRIVER') != 'dummy':
            try:
                surf.convert(32)
                surf.convert(pygame.Surface((1, 1)))
            except pygame.error:
                self.fail('convert() should not raise an exception here.')
        self.assertRaisesRegex(pygame.error, 'No video mode', surf.convert)
        pygame.display.set_mode((640, 480))
        try:
            surf.convert()
        except pygame.error:
            self.fail('convert() should not raise an exception here.')
    finally:
        pygame.display.quit()