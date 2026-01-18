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
def test_lines__arg_invalid_types(self):
    """Ensures draw lines detects invalid arg types."""
    surface = pygame.Surface((2, 2))
    color = pygame.Color('blue')
    closed = 0
    points = ((1, 2), (2, 1))
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_lines(surface, color, closed, points, '1')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_lines(surface, color, closed, (1, 2, 3))
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_lines(surface, color, InvalidBool(), points)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_lines(surface, 2.3, closed, points)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_lines((1, 2, 3, 4), color, closed, points)