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
def test_get_losses(self):
    """Ensure a surface's losses can be retrieved"""
    pygame.display.init()
    try:
        mask8 = (224, 28, 3, 0)
        mask15 = (31744, 992, 31, 0)
        mask16 = (63488, 2016, 31, 0)
        mask24 = (16711680, 65280, 255, 0)
        mask32 = (4278190080, 16711680, 65280, 255)
        display_surf = pygame.display.set_mode((100, 100))
        surf = pygame.Surface((100, 100))
        surf_8bit = pygame.Surface((100, 100), depth=8, masks=mask8)
        surf_15bit = pygame.Surface((100, 100), depth=15, masks=mask15)
        surf_16bit = pygame.Surface((100, 100), depth=16, masks=mask16)
        surf_24bit = pygame.Surface((100, 100), depth=24, masks=mask24)
        surf_32bit = pygame.Surface((100, 100), depth=32, masks=mask32)
        losses = surf.get_losses()
        self.assertIsInstance(losses, tuple)
        self.assertEqual(len(losses), 4)
        for loss in losses:
            self.assertIsInstance(loss, int)
            self.assertGreaterEqual(loss, 0)
            self.assertLessEqual(loss, 8)
        if display_surf.get_losses() == (0, 0, 0, 8):
            self.assertEqual(losses, (0, 0, 0, 8))
        elif display_surf.get_losses() == (8, 8, 8, 8):
            self.assertEqual(losses, (8, 8, 8, 8))
        self.assertEqual(surf_8bit.get_losses(), (5, 5, 6, 8))
        self.assertEqual(surf_15bit.get_losses(), (3, 3, 3, 8))
        self.assertEqual(surf_16bit.get_losses(), (3, 2, 3, 8))
        self.assertEqual(surf_24bit.get_losses(), (0, 0, 0, 8))
        self.assertEqual(surf_32bit.get_losses(), (0, 0, 0, 0))
        with self.assertRaises(pygame.error):
            surface = pygame.display.set_mode((100, 100))
            pygame.display.quit()
            surface.get_losses()
    finally:
        pygame.display.quit()