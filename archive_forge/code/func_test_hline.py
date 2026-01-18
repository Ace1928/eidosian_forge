import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_hline(self):
    """hline(surface, x1, x2, y, color): return None"""
    fg = self.foreground_color
    bg = self.background_color
    startx = 10
    stopx = 80
    y = 50
    fg_test_points = [(startx, y), (stopx, y), ((stopx - startx) // 2, y)]
    bg_test_points = [(startx - 1, y), (stopx + 1, y), (startx, y - 1), (startx, y + 1), (stopx, y - 1), (stopx, y + 1)]
    for surf in self.surfaces:
        fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
        bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
        pygame.gfxdraw.hline(surf, startx, stopx, y, fg)
        for posn in fg_test_points:
            self.check_at(surf, posn, fg_adjusted)
        for posn in bg_test_points:
            self.check_at(surf, posn, bg_adjusted)