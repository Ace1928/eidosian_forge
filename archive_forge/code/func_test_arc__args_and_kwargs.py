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
def test_arc__args_and_kwargs(self):
    """Ensures draw arc accepts a combination of args/kwargs"""
    surface = pygame.Surface((3, 1))
    color = (255, 255, 0, 0)
    rect = pygame.Rect((1, 0), (2, 3))
    start = 0.6
    stop = 2
    width = 1
    kwargs = {'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': width}
    for name in ('surface', 'color', 'rect', 'start_angle', 'stop_angle'):
        kwargs.pop(name)
        if 'surface' == name:
            bounds_rect = self.draw_arc(surface, **kwargs)
        elif 'color' == name:
            bounds_rect = self.draw_arc(surface, color, **kwargs)
        elif 'rect' == name:
            bounds_rect = self.draw_arc(surface, color, rect, **kwargs)
        elif 'start_angle' == name:
            bounds_rect = self.draw_arc(surface, color, rect, start, **kwargs)
        elif 'stop_angle' == name:
            bounds_rect = self.draw_arc(surface, color, rect, start, stop, **kwargs)
        else:
            bounds_rect = self.draw_arc(surface, color, rect, start, stop, width, **kwargs)
        self.assertIsInstance(bounds_rect, pygame.Rect)