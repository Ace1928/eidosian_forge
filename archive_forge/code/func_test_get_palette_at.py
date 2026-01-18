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
def test_get_palette_at(self):
    surf = pygame.Surface((2, 2), 0, 8)
    color = pygame.Color(1, 2, 3, 255)
    surf.set_palette_at(0, color)
    color2 = surf.get_palette_at(0)
    self.assertIsInstance(color2, pygame.Color)
    self.assertEqual(color2, color)
    self.assertRaises(IndexError, surf.get_palette_at, -1)
    self.assertRaises(IndexError, surf.get_palette_at, 256)