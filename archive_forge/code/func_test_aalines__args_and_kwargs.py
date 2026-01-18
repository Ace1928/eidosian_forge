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
def test_aalines__args_and_kwargs(self):
    """Ensures draw aalines accepts a combination of args/kwargs"""
    surface = pygame.Surface((3, 2))
    color = (255, 255, 0, 0)
    closed = 0
    points = ((1, 2), (2, 1))
    kwargs = {'surface': surface, 'color': color, 'closed': closed, 'points': points}
    for name in ('surface', 'color', 'closed', 'points'):
        kwargs.pop(name)
        if 'surface' == name:
            bounds_rect = self.draw_aalines(surface, **kwargs)
        elif 'color' == name:
            bounds_rect = self.draw_aalines(surface, color, **kwargs)
        elif 'closed' == name:
            bounds_rect = self.draw_aalines(surface, color, closed, **kwargs)
        elif 'points' == name:
            bounds_rect = self.draw_aalines(surface, color, closed, points, **kwargs)
        else:
            bounds_rect = self.draw_aalines(surface, color, closed, points, **kwargs)
        self.assertIsInstance(bounds_rect, pygame.Rect)