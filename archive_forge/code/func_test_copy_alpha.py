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
def test_copy_alpha(self):
    """issue 581: alpha of surface copy with SRCALPHA is set to 0."""
    surf = pygame.Surface((16, 16), pygame.SRCALPHA, 32)
    self.assertEqual(surf.get_alpha(), 255)
    surf2 = surf.copy()
    self.assertEqual(surf2.get_alpha(), 255)