import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def test_array_alpha(self):
    palette = [(0, 0, 0, 0), (10, 50, 100, 255), (60, 120, 240, 130), (64, 128, 255, 0), (255, 128, 0, 65)]
    targets = [self._make_src_surface(8, palette=palette), self._make_src_surface(16, palette=palette), self._make_src_surface(16, palette=palette, srcalpha=True), self._make_src_surface(24, palette=palette), self._make_src_surface(32, palette=palette), self._make_src_surface(32, palette=palette, srcalpha=True)]
    for surf in targets:
        p = palette
        if surf.get_bitsize() == 16:
            p = [surf.unmap_rgb(surf.map_rgb(c)) for c in p]
        arr = pygame.surfarray.array_alpha(surf)
        if surf.get_masks()[3]:
            for (x, y), i in self.test_points:
                self.assertEqual(arr[x, y], p[i][3], '%i != %i, posn: (%i, %i), bitsize: %i' % (arr[x, y], p[i][3], x, y, surf.get_bitsize()))
        else:
            self.assertTrue(alltrue(arr == 255))
    for surf in targets:
        blanket_alpha = surf.get_alpha()
        surf.set_alpha(None)
        arr = pygame.surfarray.array_alpha(surf)
        self.assertTrue(alltrue(arr == 255), 'All alpha values should be 255 when surf.set_alpha(None) has been set. bitsize: %i, flags: %i' % (surf.get_bitsize(), surf.get_flags()))
        surf.set_alpha(blanket_alpha)
    for surf in targets:
        blanket_alpha = surf.get_alpha()
        surf.set_alpha(0)
        arr = pygame.surfarray.array_alpha(surf)
        if surf.get_masks()[3]:
            self.assertFalse(alltrue(arr == 255), 'bitsize: %i, flags: %i' % (surf.get_bitsize(), surf.get_flags()))
        else:
            self.assertTrue(alltrue(arr == 255), 'bitsize: %i, flags: %i' % (surf.get_bitsize(), surf.get_flags()))
        surf.set_alpha(blanket_alpha)