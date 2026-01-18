import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_filled_trigon(self):
    """filled_trigon(surface, x1, y1, x2, y2, x3, y3, color): return None"""
    fg = self.foreground_color
    bg = self.background_color
    x1 = 10
    y1 = 15
    x2 = 92
    y2 = 77
    x3 = 20
    y3 = 60
    fg_test_points = [(x1, y1), (x2, y2), (x3, y3), (x1 + 10, y1 + 30)]
    bg_test_points = [(x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (x3 - 1, y3 + 1)]
    for surf in self.surfaces:
        fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
        bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
        pygame.gfxdraw.filled_trigon(surf, x1, y1, x2, y2, x3, y3, fg)
        for posn in fg_test_points:
            self.check_at(surf, posn, fg_adjusted)
        for posn in bg_test_points:
            self.check_at(surf, posn, bg_adjusted)