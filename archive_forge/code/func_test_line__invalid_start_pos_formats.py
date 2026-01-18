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
def test_line__invalid_start_pos_formats(self):
    """Ensures draw line handles invalid start_pos formats correctly."""
    kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'start_pos': None, 'end_pos': (2, 2), 'width': 1}
    start_pos_fmts = ((2,), (2, 1, 0), (2, '1'), {2, 1}, dict(((2, 1),)))
    for start_pos in start_pos_fmts:
        kwargs['start_pos'] = start_pos
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(**kwargs)