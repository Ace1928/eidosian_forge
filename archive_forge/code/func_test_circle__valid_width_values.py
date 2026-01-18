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
def test_circle__valid_width_values(self):
    """Ensures draw circle accepts different width values."""
    center = (2, 2)
    radius = 1
    pos = (center[0] - radius, center[1])
    surface_color = pygame.Color('white')
    surface = pygame.Surface((3, 4))
    color = (10, 20, 30, 255)
    kwargs = {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': None, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
    for width in (-100, -10, -1, 0, 1, 10, 100):
        surface.fill(surface_color)
        kwargs['width'] = width
        expected_color = color if width >= 0 else surface_color
        bounds_rect = self.draw_circle(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)