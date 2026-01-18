import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_ass_subscript(self):
    for bpp in (8, 16, 24, 32):
        sf = pygame.Surface((6, 8), 0, bpp)
        sf.fill((255, 255, 255))
        ar = pygame.PixelArray(sf)
        ar[..., ...] = (0, 0, 0)
        self.assertEqual(ar[0, 0], 0)
        self.assertEqual(ar[1, 0], 0)
        self.assertEqual(ar[-1, -1], 0)
        ar[...,] = (0, 0, 255)
        self.assertEqual(ar[0, 0], sf.map_rgb((0, 0, 255)))
        self.assertEqual(ar[1, 0], sf.map_rgb((0, 0, 255)))
        self.assertEqual(ar[-1, -1], sf.map_rgb((0, 0, 255)))
        ar[:, ...] = (255, 0, 0)
        self.assertEqual(ar[0, 0], sf.map_rgb((255, 0, 0)))
        self.assertEqual(ar[1, 0], sf.map_rgb((255, 0, 0)))
        self.assertEqual(ar[-1, -1], sf.map_rgb((255, 0, 0)))
        ar[...] = (0, 255, 0)
        self.assertEqual(ar[0, 0], sf.map_rgb((0, 255, 0)))
        self.assertEqual(ar[1, 0], sf.map_rgb((0, 255, 0)))
        self.assertEqual(ar[-1, -1], sf.map_rgb((0, 255, 0)))
        if hasattr(sys, 'getrefcount'):

            class Int(int):
                """Unique int instances"""
                pass
            sf = pygame.Surface((2, 2), 0, 32)
            ar = pygame.PixelArray(sf)
            x, y = (Int(0), Int(1))
            rx_before, ry_before = (sys.getrefcount(x), sys.getrefcount(y))
            ar[x, y] = 0
            rx_after, ry_after = (sys.getrefcount(x), sys.getrefcount(y))
            self.assertEqual(rx_after, rx_before)
            self.assertEqual(ry_after, ry_before)