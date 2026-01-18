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
def test_arc__valid_start_angle_values(self):
    """Ensures draw arc accepts different start_angle values."""
    expected_color = pygame.Color('blue')
    surface_color = pygame.Color('white')
    surface = pygame.Surface((6, 6))
    rect = pygame.Rect((0, 0), (4, 4))
    rect.center = surface.get_rect().center
    pos = (rect.centerx + 1, rect.centery + 1)
    kwargs = {'surface': surface, 'color': expected_color, 'rect': rect, 'start_angle': None, 'stop_angle': 17, 'width': 1}
    for start_angle in (-10.0, -5.5, -1, 0, 1, 5.5, 10.0):
        msg = f'start_angle={start_angle}'
        surface.fill(surface_color)
        kwargs['start_angle'] = start_angle
        bounds_rect = self.draw_arc(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color, msg)
        self.assertIsInstance(bounds_rect, pygame.Rect, msg)