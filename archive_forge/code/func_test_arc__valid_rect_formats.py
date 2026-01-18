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
def test_arc__valid_rect_formats(self):
    """Ensures draw arc accepts different rect formats."""
    expected_color = pygame.Color('red')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((6, 6))
    rect = pygame.Rect((0, 0), (4, 4))
    rect.center = surface.get_rect().center
    pos = (rect.centerx + 1, rect.centery + 1)
    kwargs = {'surface': surface, 'color': expected_color, 'rect': None, 'start_angle': 0, 'stop_angle': 7, 'width': 1}
    rects = (rect, (rect.topleft, rect.size), (rect.x, rect.y, rect.w, rect.h))
    for rect in rects:
        surface.fill(surface_color)
        kwargs['rect'] = rect
        bounds_rect = self.draw_arc(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)