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
def test_aaline__kwarg_invalid_name(self):
    """Ensures draw aaline detects invalid kwarg names."""
    surface = pygame.Surface((2, 3))
    color = pygame.Color('cyan')
    start_pos = (1, 1)
    end_pos = (2, 0)
    kwargs_list = [{'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'invalid': 1}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'invalid': 1}]
    for kwargs in kwargs_list:
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(**kwargs)