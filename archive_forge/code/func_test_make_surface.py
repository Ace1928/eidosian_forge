import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_make_surface(self):
    bg_color = pygame.Color(255, 255, 255)
    fg_color = pygame.Color(128, 100, 0)
    for bpp in (8, 16, 24, 32):
        sf = pygame.Surface((10, 20), 0, bpp)
        bg_color_adj = sf.unmap_rgb(sf.map_rgb(bg_color))
        fg_color_adj = sf.unmap_rgb(sf.map_rgb(fg_color))
        sf.fill(bg_color_adj)
        sf.fill(fg_color_adj, (2, 5, 4, 11))
        ar = pygame.PixelArray(sf)
        newsf = ar[::2, ::2].make_surface()
        rect = newsf.get_rect()
        self.assertEqual(rect.width, 5)
        self.assertEqual(rect.height, 10)
        for p in [(0, 2), (0, 3), (1, 2), (2, 2), (3, 2), (3, 3), (0, 7), (0, 8), (1, 8), (2, 8), (3, 8), (3, 7)]:
            self.assertEqual(newsf.get_at(p), bg_color_adj)
        for p in [(1, 3), (2, 3), (1, 5), (2, 5), (1, 7), (2, 7)]:
            self.assertEqual(newsf.get_at(p), fg_color_adj)
    w = 17
    lst = list(range(w))
    w_slice = len(lst[::2])
    h = 3
    sf = pygame.Surface((w, h), 0, 32)
    ar = pygame.PixelArray(sf)
    ar2 = ar[::2, :]
    sf2 = ar2.make_surface()
    w2, h2 = sf2.get_size()
    self.assertEqual(w2, w_slice)
    self.assertEqual(h2, h)
    h = 17
    lst = list(range(h))
    h_slice = len(lst[::2])
    w = 3
    sf = pygame.Surface((w, h), 0, 32)
    ar = pygame.PixelArray(sf)
    ar2 = ar[:, ::2]
    sf2 = ar2.make_surface()
    w2, h2 = sf2.get_size()
    self.assertEqual(w2, w)
    self.assertEqual(h2, h_slice)