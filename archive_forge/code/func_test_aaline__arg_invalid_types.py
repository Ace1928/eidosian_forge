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
def test_aaline__arg_invalid_types(self):
    """Ensures draw aaline detects invalid arg types."""
    surface = pygame.Surface((2, 2))
    color = pygame.Color('blue')
    start_pos = (0, 1)
    end_pos = (1, 2)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_aaline(surface, color, start_pos, (1, 2, 3))
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_aaline(surface, color, (1,), end_pos)
    with self.assertRaises(ValueError):
        bounds_rect = self.draw_aaline(surface, 'invalid-color', start_pos, end_pos)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_aaline((1, 2, 3, 4), color, start_pos, end_pos)