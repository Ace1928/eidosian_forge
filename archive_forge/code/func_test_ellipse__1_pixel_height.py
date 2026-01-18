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
def test_ellipse__1_pixel_height(self):
    """Ensures an ellipse with a height of 1 is drawn correctly.

        An ellipse with a height of 1 pixel is a horizontal line.
        """
    ellipse_color = pygame.Color('red')
    surface_color = pygame.Color('black')
    surf_w, surf_h = (20, 10)
    surface = pygame.Surface((surf_w, surf_h))
    rect = pygame.Rect((0, 0), (0, 1))
    collide_rect = rect.copy()
    off_right = surf_w
    off_top = -1
    off_bottom = surf_h
    center_x = surf_w // 2
    center_y = surf_h // 2
    for ellipse_w in range(6, 10):
        collide_rect.w = ellipse_w
        rect.w = ellipse_w
        off_left = -(ellipse_w + 1)
        half_off_left = -(ellipse_w // 2)
        half_off_right = surf_w - ellipse_w // 2
        positions = ((off_left, off_top), (half_off_left, off_top), (center_x, off_top), (half_off_right, off_top), (off_right, off_top), (off_left, center_y), (half_off_left, center_y), (center_x, center_y), (half_off_right, center_y), (off_right, center_y), (off_left, off_bottom), (half_off_left, off_bottom), (center_x, off_bottom), (half_off_right, off_bottom), (off_right, off_bottom))
        for rect_pos in positions:
            surface.fill(surface_color)
            rect.topleft = rect_pos
            collide_rect.topleft = rect_pos
            self.draw_ellipse(surface, ellipse_color, rect)
            self._check_1_pixel_sized_ellipse(surface, collide_rect, surface_color, ellipse_color)