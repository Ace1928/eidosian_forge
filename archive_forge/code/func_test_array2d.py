import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def test_array2d(self):
    sources = [self._make_src_surface(8), self._make_src_surface(16), self._make_src_surface(16, srcalpha=True), self._make_src_surface(24), self._make_src_surface(32), self._make_src_surface(32, srcalpha=True)]
    palette = self.test_palette
    alpha_color = (0, 0, 0, 128)
    for surf in sources:
        arr = pygame.surfarray.array2d(surf)
        for posn, i in self.test_points:
            self.assertEqual(arr[posn], surf.get_at_mapped(posn), '%s != %s: flags: %i, bpp: %i, posn: %s' % (arr[posn], surf.get_at_mapped(posn), surf.get_flags(), surf.get_bitsize(), posn))
        if surf.get_masks()[3]:
            surf.fill(alpha_color)
            arr = pygame.surfarray.array2d(surf)
            posn = (0, 0)
            self.assertEqual(arr[posn], surf.get_at_mapped(posn), '%s != %s: bpp: %i' % (arr[posn], surf.get_at_mapped(posn), surf.get_bitsize()))