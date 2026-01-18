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
def test_ellipse__1_pixel_width(self):
    """Ensures an ellipse with a width of 1 is drawn correctly.

        An ellipse with a width of 1 pixel is a vertical line.
        """
    ellipse_color = pygame.Color('red')
    surface_color = pygame.Color('black')
    surf_w, surf_h = (10, 20)
    surface = pygame.Surface((surf_w, surf_h))
    rect = pygame.Rect((0, 0), (1, 0))
    collide_rect = rect.copy()
    off_left = -1
    off_right = surf_w
    off_bottom = surf_h
    center_x = surf_w // 2
    center_y = surf_h // 2
    for ellipse_h in range(6, 10):
        collide_rect.h = ellipse_h
        rect.h = ellipse_h
        off_top = -(ellipse_h + 1)
        half_off_top = -(ellipse_h // 2)
        half_off_bottom = surf_h - ellipse_h // 2
        positions = ((off_left, off_top), (off_left, half_off_top), (off_left, center_y), (off_left, half_off_bottom), (off_left, off_bottom), (center_x, off_top), (center_x, half_off_top), (center_x, center_y), (center_x, half_off_bottom), (center_x, off_bottom), (off_right, off_top), (off_right, half_off_top), (off_right, center_y), (off_right, half_off_bottom), (off_right, off_bottom))
        for rect_pos in positions:
            surface.fill(surface_color)
            rect.topleft = rect_pos
            collide_rect.topleft = rect_pos
            self.draw_ellipse(surface, ellipse_color, rect)
            self._check_1_pixel_sized_ellipse(surface, collide_rect, surface_color, ellipse_color)