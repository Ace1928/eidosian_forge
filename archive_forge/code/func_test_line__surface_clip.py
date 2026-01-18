import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
def test_line__surface_clip(self):
    """Ensures draw line respects a surface's clip area."""
    surfw = surfh = 30
    line_color = pygame.Color('red')
    surface_color = pygame.Color('green')
    surface = pygame.Surface((surfw, surfh))
    surface.fill(surface_color)
    clip_rect = pygame.Rect((0, 0), (11, 11))
    clip_rect.center = surface.get_rect().center
    pos_rect = clip_rect.copy()
    for thickness in (1, 3):
        for center in rect_corners_mids_and_center(clip_rect):
            pos_rect.center = center
            surface.set_clip(None)
            surface.fill(surface_color)
            self.draw_line(surface, line_color, pos_rect.midtop, pos_rect.midbottom, thickness)
            expected_pts = get_color_points(surface, line_color, clip_rect)
            surface.fill(surface_color)
            surface.set_clip(clip_rect)
            self.draw_line(surface, line_color, pos_rect.midtop, pos_rect.midbottom, thickness)
            surface.lock()
            for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                if pt in expected_pts:
                    expected_color = line_color
                else:
                    expected_color = surface_color
                self.assertEqual(surface.get_at(pt), expected_color, pt)
            surface.unlock()