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
def test_circle_shape(self):
    """Ensures there are no holes in the circle, and no overdrawing.

        Tests drawing a thick circle.
        Measures the distance of the drawn pixels from the circle center.
        """
    surfw = surfh = 100
    circle_color = pygame.Color('red')
    surface_color = pygame.Color('green')
    surface = pygame.Surface((surfw, surfh))
    surface.fill(surface_color)
    cx, cy = center = (50, 50)
    radius = 45
    width = 25
    dest_rect = self.draw_circle(surface, circle_color, center, radius, width)
    for pt in test_utils.rect_area_pts(dest_rect):
        x, y = pt
        sqr_distance = (x - cx) ** 2 + (y - cy) ** 2
        if (radius - width + 1) ** 2 < sqr_distance < (radius - 1) ** 2:
            self.assertEqual(surface.get_at(pt), circle_color)
        if sqr_distance < (radius - width - 1) ** 2 or sqr_distance > (radius + 1) ** 2:
            self.assertEqual(surface.get_at(pt), surface_color)