import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_filled_circle(self):
    """filled_circle(surface, x, y, r, color): return None"""
    fg = self.foreground_color
    bg = self.background_color
    x = 45
    y = 40
    r = 30
    fg_test_points = [(x, y - r), (x, y - r + 1), (x, y + r), (x, y + r - 1), (x - r, y), (x - r + 1, y), (x + r, y), (x + r - 1, y), (x, y)]
    bg_test_points = [(x, y - r - 1), (x, y + r + 1), (x - r - 1, y), (x + r + 1, y)]
    for surf in self.surfaces:
        fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
        bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
        pygame.gfxdraw.filled_circle(surf, x, y, r, fg)
        for posn in fg_test_points:
            self.check_at(surf, posn, fg_adjusted)
        for posn in bg_test_points:
            self.check_at(surf, posn, bg_adjusted)