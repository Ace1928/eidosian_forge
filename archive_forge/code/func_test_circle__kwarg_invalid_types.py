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
def test_circle__kwarg_invalid_types(self):
    """Ensures draw circle detects invalid kwarg types."""
    surface = pygame.Surface((3, 3))
    color = pygame.Color('green')
    center = (0, 1)
    radius = 1
    width = 1
    quadrant = 1
    kwargs_list = [{'surface': pygame.Surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': 2.3, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': (1, 1, 1), 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': '1', 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': 1.2, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': 'True', 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': 'True', 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': 3.14, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': 'quadrant'}]
    for kwargs in kwargs_list:
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(**kwargs)