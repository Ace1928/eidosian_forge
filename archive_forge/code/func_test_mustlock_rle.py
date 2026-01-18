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
def test_mustlock_rle(self):
    """Test RLEACCEL flag with mustlock()"""
    surf = pygame.Surface((100, 100))
    blit_surf = pygame.Surface((100, 100))
    blit_surf.set_colorkey((0, 0, 255), pygame.RLEACCEL)
    self.assertTrue(blit_surf.get_flags() & pygame.RLEACCELOK)
    surf.blit(blit_surf, (0, 0))
    self.assertTrue(blit_surf.get_flags() & pygame.RLEACCEL)
    self.assertTrue(blit_surf.mustlock())