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
def test_arc__valid_color_formats(self):
    """Ensures draw arc accepts different color formats."""
    green_color = pygame.Color('green')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((6, 6))
    rect = pygame.Rect((0, 0), (4, 4))
    rect.center = surface.get_rect().center
    pos = (rect.centerx + 1, rect.centery + 1)
    kwargs = {'surface': surface, 'color': None, 'rect': rect, 'start_angle': 0, 'stop_angle': 7, 'width': 1}
    greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
    for color in greens:
        surface.fill(surface_color)
        kwargs['color'] = color
        if isinstance(color, int):
            expected_color = surface.unmap_rgb(color)
        else:
            expected_color = green_color
        bounds_rect = self.draw_arc(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)