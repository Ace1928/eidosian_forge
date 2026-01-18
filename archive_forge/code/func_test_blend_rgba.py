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
def test_blend_rgba(self):
    bitsizes = [16, 32]
    blends = ['BLEND_RGBA_ADD', 'BLEND_RGBA_SUB', 'BLEND_RGBA_MULT', 'BLEND_RGBA_MIN', 'BLEND_RGBA_MAX']
    for bitsize in bitsizes:
        surf = self._make_surface(bitsize, srcalpha=True)
        comp = self._make_surface(bitsize, srcalpha=True)
        for blend in blends:
            self._fill_surface(surf)
            self._fill_surface(comp)
            comp.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
            surf.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
            self._assert_same(surf, comp)