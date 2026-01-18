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
def test_ellipse__arg_invalid_types(self):
    """Ensures draw ellipse detects invalid arg types."""
    surface = pygame.Surface((2, 2))
    color = pygame.Color('blue')
    rect = pygame.Rect((1, 1), (1, 1))
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_ellipse(surface, color, rect, '1')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_ellipse(surface, color, (1, 2, 3, 4, 5), 1)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_ellipse(surface, 2.3, rect, 0)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_ellipse(rect, color, rect, 2)