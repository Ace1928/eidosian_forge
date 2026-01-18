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
def test_aalines__valid_closed_values(self):
    """Ensures draw aalines accepts different closed values."""
    line_color = pygame.Color('blue')
    surface_color = pygame.Color('white')
    surface = pygame.Surface((5, 5))
    pos = (1, 3)
    kwargs = {'surface': surface, 'color': line_color, 'closed': None, 'points': ((1, 1), (4, 1), (4, 4), (1, 4))}
    true_values = (-7, 1, 10, '2', 3.1, (4,), [5], True)
    false_values = (None, '', 0, (), [], False)
    for closed in true_values + false_values:
        surface.fill(surface_color)
        kwargs['closed'] = closed
        expected_color = line_color if closed else surface_color
        bounds_rect = self.draw_aalines(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)