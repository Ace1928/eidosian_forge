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
def test_line__kwarg_invalid_types(self):
    """Ensures draw line detects invalid kwarg types."""
    surface = pygame.Surface((3, 3))
    color = pygame.Color('green')
    start_pos = (1, 0)
    end_pos = (2, 0)
    width = 1
    kwargs_list = [{'surface': pygame.Surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': width}, {'surface': surface, 'color': 2.3, 'start_pos': start_pos, 'end_pos': end_pos, 'width': width}, {'surface': surface, 'color': color, 'start_pos': (0, 0, 0), 'end_pos': end_pos, 'width': width}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': (0,), 'width': width}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': 1.2}]
    for kwargs in kwargs_list:
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(**kwargs)