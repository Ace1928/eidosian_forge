import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def test_array_colorkey(self):
    palette = [(0, 0, 0, 0), (10, 50, 100, 255), (60, 120, 240, 130), (64, 128, 255, 0), (255, 128, 0, 65)]
    targets = [self._make_src_surface(8, palette=palette), self._make_src_surface(16, palette=palette), self._make_src_surface(16, palette=palette, srcalpha=True), self._make_src_surface(24, palette=palette), self._make_src_surface(32, palette=palette), self._make_src_surface(32, palette=palette, srcalpha=True)]
    for surf in targets:
        p = palette
        if surf.get_bitsize() == 16:
            p = [surf.unmap_rgb(surf.map_rgb(c)) for c in p]
        surf.set_colorkey(None)
        arr = pygame.surfarray.array_colorkey(surf)
        self.assertTrue(alltrue(arr == 255))
        for i in range(1, len(palette)):
            surf.set_colorkey(p[i])
            alphas = [255] * len(p)
            alphas[i] = 0
            arr = pygame.surfarray.array_colorkey(surf)
            for (x, y), j in self.test_points:
                self.assertEqual(arr[x, y], alphas[j], '%i != %i, posn: (%i, %i), bitsize: %i' % (arr[x, y], alphas[j], x, y, surf.get_bitsize()))