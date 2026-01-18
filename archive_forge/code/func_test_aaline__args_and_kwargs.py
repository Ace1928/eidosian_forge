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
def test_aaline__args_and_kwargs(self):
    """Ensures draw aaline accepts a combination of args/kwargs"""
    surface = pygame.Surface((3, 2))
    color = (255, 255, 0, 0)
    start_pos = (0, 1)
    end_pos = (1, 2)
    kwargs = {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos}
    for name in ('surface', 'color', 'start_pos', 'end_pos'):
        kwargs.pop(name)
        if 'surface' == name:
            bounds_rect = self.draw_aaline(surface, **kwargs)
        elif 'color' == name:
            bounds_rect = self.draw_aaline(surface, color, **kwargs)
        elif 'start_pos' == name:
            bounds_rect = self.draw_aaline(surface, color, start_pos, **kwargs)
        elif 'end_pos' == name:
            bounds_rect = self.draw_aaline(surface, color, start_pos, end_pos, **kwargs)
        else:
            bounds_rect = self.draw_aaline(surface, color, start_pos, end_pos, **kwargs)
        self.assertIsInstance(bounds_rect, pygame.Rect)