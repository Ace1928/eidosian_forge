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
def test_1_pixel_high_or_wide_shapes(self):
    pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
    self.draw_polygon(self.surface, GREEN, [(x, 2) for x, _y in CROSS], 0)
    cross_size = 6
    for x in range(cross_size + 1):
        self.assertEqual(self.surface.get_at((x, 1)), RED)
        self.assertEqual(self.surface.get_at((x, 2)), GREEN)
        self.assertEqual(self.surface.get_at((x, 3)), RED)
    pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
    self.draw_polygon(self.surface, GREEN, [(x, 5) for x, _y in CROSS], 1)
    for x in range(cross_size + 1):
        self.assertEqual(self.surface.get_at((x, 4)), RED)
        self.assertEqual(self.surface.get_at((x, 5)), GREEN)
        self.assertEqual(self.surface.get_at((x, 6)), RED)
    pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
    self.draw_polygon(self.surface, GREEN, [(3, y) for _x, y in CROSS], 0)
    for y in range(cross_size + 1):
        self.assertEqual(self.surface.get_at((2, y)), RED)
        self.assertEqual(self.surface.get_at((3, y)), GREEN)
        self.assertEqual(self.surface.get_at((4, y)), RED)
    pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
    self.draw_polygon(self.surface, GREEN, [(4, y) for _x, y in CROSS], 1)
    for y in range(cross_size + 1):
        self.assertEqual(self.surface.get_at((3, y)), RED)
        self.assertEqual(self.surface.get_at((4, y)), GREEN)
        self.assertEqual(self.surface.get_at((5, y)), RED)