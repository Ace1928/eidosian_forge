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
def test_mustlock_surf_alpha_rle(self):
    """Test RLEACCEL flag with mustlock() on a surface
        with per pixel alpha - new feature in SDL2"""
    surf = pygame.Surface((100, 100))
    blit_surf = pygame.Surface((100, 100), depth=32, flags=pygame.SRCALPHA)
    blit_surf.set_colorkey((192, 191, 192, 255), pygame.RLEACCEL)
    self.assertTrue(blit_surf.get_flags() & pygame.RLEACCELOK)
    surf.blit(blit_surf, (0, 0))
    self.assertTrue(blit_surf.get_flags() & pygame.RLEACCEL)
    self.assertTrue(blit_surf.get_flags() & pygame.SRCALPHA)
    self.assertTrue(blit_surf.mustlock())