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
def test_ellipse__1_pixel_width_and_height(self):
    """Ensures an ellipse with a width and height of 1 is drawn correctly.

        An ellipse with a width and height of 1 pixel is a single pixel.
        """
    ellipse_color = pygame.Color('red')
    surface_color = pygame.Color('black')
    surf_w, surf_h = (10, 10)
    surface = pygame.Surface((surf_w, surf_h))
    rect = pygame.Rect((0, 0), (1, 1))
    off_left = -1
    off_right = surf_w
    off_top = -1
    off_bottom = surf_h
    left_edge = 0
    right_edge = surf_w - 1
    top_edge = 0
    bottom_edge = surf_h - 1
    center_x = surf_w // 2
    center_y = surf_h // 2
    positions = ((off_left, off_top), (off_left, top_edge), (off_left, center_y), (off_left, bottom_edge), (off_left, off_bottom), (left_edge, off_top), (left_edge, top_edge), (left_edge, center_y), (left_edge, bottom_edge), (left_edge, off_bottom), (center_x, off_top), (center_x, top_edge), (center_x, center_y), (center_x, bottom_edge), (center_x, off_bottom), (right_edge, off_top), (right_edge, top_edge), (right_edge, center_y), (right_edge, bottom_edge), (right_edge, off_bottom), (off_right, off_top), (off_right, top_edge), (off_right, center_y), (off_right, bottom_edge), (off_right, off_bottom))
    for rect_pos in positions:
        surface.fill(surface_color)
        rect.topleft = rect_pos
        self.draw_ellipse(surface, ellipse_color, rect)
        self._check_1_pixel_sized_ellipse(surface, rect, surface_color, ellipse_color)