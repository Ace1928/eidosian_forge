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
def test_lines__valid_width_values(self):
    """Ensures draw lines accepts different width values."""
    line_color = pygame.Color('yellow')
    surface_color = pygame.Color('white')
    surface = pygame.Surface((3, 4))
    pos = (1, 1)
    kwargs = {'surface': surface, 'color': line_color, 'closed': False, 'points': (pos, (2, 1)), 'width': None}
    for width in (-100, -10, -1, 0, 1, 10, 100):
        surface.fill(surface_color)
        kwargs['width'] = width
        expected_color = line_color if width > 0 else surface_color
        bounds_rect = self.draw_lines(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)