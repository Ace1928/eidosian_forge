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
def test_aalines__kwarg_invalid_types(self):
    """Ensures draw aalines detects invalid kwarg types."""
    valid_kwargs = {'surface': pygame.Surface((3, 3)), 'color': pygame.Color('green'), 'closed': False, 'points': ((1, 2), (2, 1))}
    invalid_kwargs = {'surface': pygame.Surface, 'color': 2.3, 'closed': InvalidBool(), 'points': (0, 0, 0)}
    for kwarg in ('surface', 'color', 'closed', 'points'):
        kwargs = dict(valid_kwargs)
        kwargs[kwarg] = invalid_kwargs[kwarg]
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(**kwargs)