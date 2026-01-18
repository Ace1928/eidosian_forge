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
def test_mustlock(self):
    surf = pygame.Surface((1024, 1024))
    subsurf = surf.subsurface((0, 0, 1024, 1024))
    self.assertTrue(subsurf.mustlock())
    self.assertFalse(surf.mustlock())
    rects = ((0, 0, 512, 512), (0, 0, 256, 256), (0, 0, 128, 128))
    surf_stack = []
    surf_stack.append(surf)
    surf_stack.append(subsurf)
    for rect in rects:
        surf_stack.append(surf_stack[-1].subsurface(rect))
        self.assertTrue(surf_stack[-1].mustlock())
        self.assertTrue(surf_stack[-2].mustlock())