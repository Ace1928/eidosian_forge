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
def test_aaline__kwargs_missing(self):
    """Ensures draw aaline detects any missing required kwargs."""
    kwargs = {'surface': pygame.Surface((3, 2)), 'color': pygame.Color('red'), 'start_pos': (2, 1), 'end_pos': (2, 2)}
    for name in ('end_pos', 'start_pos', 'color', 'surface'):
        invalid_kwargs = dict(kwargs)
        invalid_kwargs.pop(name)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(**invalid_kwargs)