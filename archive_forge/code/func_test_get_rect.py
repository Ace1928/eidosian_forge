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
def test_get_rect(self):
    """Ensure a surface's rect can be retrieved."""
    size = (16, 16)
    surf = pygame.Surface(size)
    rect = surf.get_rect()
    self.assertEqual(rect.size, size)