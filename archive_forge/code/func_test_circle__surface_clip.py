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
def test_circle__surface_clip(self):
    """Ensures draw circle respects a surface's clip area.

        Tests drawing the circle filled and unfilled.
        """
    surfw = surfh = 25
    circle_color = pygame.Color('red')
    surface_color = pygame.Color('green')
    surface = pygame.Surface((surfw, surfh))
    surface.fill(surface_color)
    clip_rect = pygame.Rect((0, 0), (10, 10))
    clip_rect.center = surface.get_rect().center
    radius = clip_rect.w // 2 + 1
    for width in (0, 1):
        for center in rect_corners_mids_and_center(clip_rect):
            surface.set_clip(None)
            surface.fill(surface_color)
            self.draw_circle(surface, circle_color, center, radius, width)
            expected_pts = get_color_points(surface, circle_color, clip_rect)
            surface.fill(surface_color)
            surface.set_clip(clip_rect)
            self.draw_circle(surface, circle_color, center, radius, width)
            surface.lock()
            for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                if pt in expected_pts:
                    expected_color = circle_color
                else:
                    expected_color = surface_color
                self.assertEqual(surface.get_at(pt), expected_color, pt)
            surface.unlock()