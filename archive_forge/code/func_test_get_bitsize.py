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
def test_get_bitsize(self):
    pygame.display.init()
    try:
        expected_size = (11, 21)
        expected_depth = 32
        surface = pygame.Surface(expected_size, pygame.SRCALPHA, expected_depth)
        self.assertEqual(surface.get_size(), expected_size)
        self.assertEqual(surface.get_bitsize(), expected_depth)
        expected_depth = 16
        surface = pygame.Surface(expected_size, pygame.SRCALPHA, expected_depth)
        self.assertEqual(surface.get_size(), expected_size)
        self.assertEqual(surface.get_bitsize(), expected_depth)
        expected_depth = 15
        surface = pygame.Surface(expected_size, 0, expected_depth)
        self.assertEqual(surface.get_size(), expected_size)
        self.assertEqual(surface.get_bitsize(), expected_depth)
        expected_depth = -1
        self.assertRaises(ValueError, pygame.Surface, expected_size, 0, expected_depth)
        expected_depth = 11
        self.assertRaises(ValueError, pygame.Surface, expected_size, 0, expected_depth)
        expected_depth = 1024
        self.assertRaises(ValueError, pygame.Surface, expected_size, 0, expected_depth)
        with self.assertRaises(pygame.error):
            surface = pygame.display.set_mode()
            pygame.display.quit()
            surface.get_bitsize()
    finally:
        pygame.display.quit()