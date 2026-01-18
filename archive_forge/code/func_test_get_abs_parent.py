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
def test_get_abs_parent(self):
    pygame.display.init()
    try:
        parent = pygame.Surface((32, 32), SRCALPHA, 32)
        sub_level_1 = parent.subsurface((1, 1), (15, 15))
        sub_level_2 = sub_level_1.subsurface((1, 1), (12, 12))
        sub_level_3 = sub_level_2.subsurface((1, 1), (9, 9))
        sub_level_4 = sub_level_3.subsurface((1, 1), (8, 8))
        sub_level_5 = sub_level_4.subsurface((2, 2), (3, 4))
        sub_level_6 = sub_level_5.subsurface((0, 0), (2, 1))
        self.assertRaises(ValueError, parent.subsurface, (5, 5), (100, 100))
        self.assertRaises(ValueError, sub_level_3.subsurface, (0, 0), (11, 5))
        self.assertRaises(ValueError, sub_level_6.subsurface, (0, 0), (5, 5))
        self.assertEqual(parent.get_abs_parent(), parent)
        self.assertEqual(sub_level_1.get_abs_parent(), sub_level_1.get_parent())
        self.assertEqual(sub_level_2.get_abs_parent(), parent)
        self.assertEqual(sub_level_3.get_abs_parent(), parent)
        self.assertEqual(sub_level_4.get_abs_parent(), parent)
        self.assertEqual(sub_level_5.get_abs_parent(), parent)
        self.assertEqual(sub_level_6.get_abs_parent(), sub_level_6.get_parent().get_abs_parent())
        with self.assertRaises(pygame.error):
            surface = pygame.display.set_mode()
            pygame.display.quit()
            surface.get_abs_parent()
    finally:
        pygame.display.quit()