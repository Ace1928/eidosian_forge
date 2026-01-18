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
def test_arc__valid_width_values(self):
    """Ensures draw arc accepts different width values."""
    arc_color = pygame.Color('yellow')
    surface_color = pygame.Color('white')
    surface = pygame.Surface((6, 6))
    rect = pygame.Rect((0, 0), (4, 4))
    rect.center = surface.get_rect().center
    pos = (rect.centerx + 1, rect.centery + 1)
    kwargs = {'surface': surface, 'color': arc_color, 'rect': rect, 'start_angle': 0, 'stop_angle': 7, 'width': None}
    for width in (-50, -10, -3, -2, -1, 0, 1, 2, 3, 10, 50):
        msg = f'width={width}'
        surface.fill(surface_color)
        kwargs['width'] = width
        expected_color = arc_color if width > 0 else surface_color
        bounds_rect = self.draw_arc(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color, msg)
        self.assertIsInstance(bounds_rect, pygame.Rect, msg)