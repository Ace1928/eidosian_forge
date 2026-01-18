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
def test_get_shifts(self):
    """
        Tests whether Surface.get_shifts returns proper
        RGBA shifts under various conditions.
        """
    depths = [8, 24, 32]
    alpha = 128
    off = None
    for bit_depth in depths:
        surface = pygame.Surface((32, 32), depth=bit_depth)
        surface.set_alpha(alpha)
        r1, g1, b1, a1 = surface.get_shifts()
        surface.set_alpha(off)
        r2, g2, b2, a2 = surface.get_shifts()
        self.assertEqual((r1, g1, b1, a1), (r2, g2, b2, a2))