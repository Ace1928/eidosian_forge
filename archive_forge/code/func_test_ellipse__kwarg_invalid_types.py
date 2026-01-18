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
def test_ellipse__kwarg_invalid_types(self):
    """Ensures draw ellipse detects invalid kwarg types."""
    surface = pygame.Surface((3, 3))
    color = pygame.Color('green')
    rect = pygame.Rect((0, 1), (1, 1))
    kwargs_list = [{'surface': pygame.Surface, 'color': color, 'rect': rect, 'width': 1}, {'surface': surface, 'color': 2.3, 'rect': rect, 'width': 1}, {'surface': surface, 'color': color, 'rect': (0, 0, 0), 'width': 1}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1.1}]
    for kwargs in kwargs_list:
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(**kwargs)