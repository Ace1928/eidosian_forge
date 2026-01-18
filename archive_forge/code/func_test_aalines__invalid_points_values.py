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
def test_aalines__invalid_points_values(self):
    """Ensures draw aalines handles invalid points values correctly."""
    kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'closed': False, 'points': None}
    for points in ([], ((1, 1),)):
        for seq_type in (tuple, list):
            kwargs['points'] = seq_type(points)
            with self.assertRaises(ValueError):
                bounds_rect = self.draw_aalines(**kwargs)