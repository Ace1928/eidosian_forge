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
def test_get_at_mapped(self):
    color = pygame.Color(10, 20, 30)
    for bitsize in [8, 16, 24, 32]:
        surf = pygame.Surface((2, 2), 0, bitsize)
        surf.fill(color)
        pixel = surf.get_at_mapped((0, 0))
        self.assertEqual(pixel, surf.map_rgb(color), '%i != %i, bitsize: %i' % (pixel, surf.map_rgb(color), bitsize))