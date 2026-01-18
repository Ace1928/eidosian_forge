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
def test_SRCALPHA(self):
    surf = pygame.Surface((70, 70), SRCALPHA, 32)
    self.assertEqual(surf.get_flags() & SRCALPHA, SRCALPHA)
    self.assertRaises(ValueError, pygame.Surface, (100, 100), pygame.SRCALPHA, 24)
    surf2 = pygame.Surface((70, 70), SRCALPHA)
    if surf2.get_bitsize() == 32:
        self.assertEqual(surf2.get_flags() & SRCALPHA, SRCALPHA)