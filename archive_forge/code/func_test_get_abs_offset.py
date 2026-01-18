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
def test_get_abs_offset(self):
    pygame.display.init()
    try:
        parent = pygame.Surface((64, 64), SRCALPHA, 32)
        sub_level_1 = parent.subsurface((2, 2), (34, 37))
        sub_level_2 = sub_level_1.subsurface((0, 0), (30, 29))
        sub_level_3 = sub_level_2.subsurface((3, 7), (20, 21))
        sub_level_4 = sub_level_3.subsurface((6, 1), (14, 14))
        sub_level_5 = sub_level_4.subsurface((5, 6), (3, 4))
        self.assertEqual(parent.get_abs_offset(), (0, 0))
        self.assertEqual(sub_level_1.get_abs_offset(), (2, 2))
        self.assertEqual(sub_level_2.get_abs_offset(), (2, 2))
        self.assertEqual(sub_level_3.get_abs_offset(), (5, 9))
        self.assertEqual(sub_level_4.get_abs_offset(), (11, 10))
        self.assertEqual(sub_level_5.get_abs_offset(), (16, 16))
        with self.assertRaises(pygame.error):
            surface = pygame.display.set_mode()
            pygame.display.quit()
            surface.get_abs_offset()
    finally:
        pygame.display.quit()