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
def test_copy_rle(self):
    """Test copying a surface set to use run length encoding"""
    s1 = pygame.Surface((32, 32), 24)
    s1.set_colorkey((255, 0, 255), pygame.RLEACCEL)
    self.assertTrue(s1.get_flags() & pygame.RLEACCELOK)
    newsurf = s1.copy()
    self.assertTrue(s1.get_flags() & pygame.RLEACCELOK)
    self.assertTrue(newsurf.get_flags() & pygame.RLEACCELOK)