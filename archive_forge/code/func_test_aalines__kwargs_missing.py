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
def test_aalines__kwargs_missing(self):
    """Ensures draw aalines detects any missing required kwargs."""
    kwargs = {'surface': pygame.Surface((3, 2)), 'color': pygame.Color('red'), 'closed': 1, 'points': ((2, 2), (1, 1))}
    for name in ('points', 'closed', 'color', 'surface'):
        invalid_kwargs = dict(kwargs)
        invalid_kwargs.pop(name)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(**invalid_kwargs)