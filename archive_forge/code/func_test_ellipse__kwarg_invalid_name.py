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
def test_ellipse__kwarg_invalid_name(self):
    """Ensures draw ellipse detects invalid kwarg names."""
    surface = pygame.Surface((2, 3))
    color = pygame.Color('cyan')
    rect = pygame.Rect((0, 1), (2, 2))
    kwargs_list = [{'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'rect': rect, 'invalid': 1}]
    for kwargs in kwargs_list:
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(**kwargs)