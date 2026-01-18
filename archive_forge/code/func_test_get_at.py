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
def test_get_at(self):
    surf = pygame.Surface((2, 2), 0, 24)
    c00 = pygame.Color(1, 2, 3)
    c01 = pygame.Color(5, 10, 15)
    c10 = pygame.Color(100, 50, 0)
    c11 = pygame.Color(4, 5, 6)
    surf.set_at((0, 0), c00)
    surf.set_at((0, 1), c01)
    surf.set_at((1, 0), c10)
    surf.set_at((1, 1), c11)
    c = surf.get_at((0, 0))
    self.assertIsInstance(c, pygame.Color)
    self.assertEqual(c, c00)
    self.assertEqual(surf.get_at((0, 1)), c01)
    self.assertEqual(surf.get_at((1, 0)), c10)
    self.assertEqual(surf.get_at((1, 1)), c11)
    for p in [(-1, 0), (0, -1), (2, 0), (0, 2)]:
        self.assertRaises(IndexError, surf.get_at, p)