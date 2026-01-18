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
def test_anti_aliasing_float_coordinates(self):
    """Float coordinates should be blended smoothly."""
    self.surface = pygame.Surface((10, 10))
    draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
    check_points = [(i, j) for i in range(5) for j in range(5)]
    brown = (127, 127, 0)
    reddish = (191, 63, 0)
    greenish = (63, 191, 0)
    expected = {(2, 2): FG_GREEN}
    self._check_antialiasing((1.5, 2), (1.5, 2), expected, check_points, set_endpoints=False)
    expected = {(2, 3): FG_GREEN}
    self._check_antialiasing((2.49, 2.7), (2.49, 2.7), expected, check_points, set_endpoints=False)
    expected = {(1, 2): brown, (2, 2): FG_GREEN}
    self._check_antialiasing((1.5, 2), (2, 2), expected, check_points, set_endpoints=False)
    expected = {(1, 2): brown, (2, 2): FG_GREEN, (3, 2): brown}
    self._check_antialiasing((1.5, 2), (2.5, 2), expected, check_points, set_endpoints=False)
    expected = {(2, 2): brown, (1, 2): FG_GREEN}
    self._check_antialiasing((1, 2), (1.5, 2), expected, check_points, set_endpoints=False)
    expected = {(1, 2): brown, (2, 2): greenish}
    self._check_antialiasing((1.5, 2), (1.75, 2), expected, check_points, set_endpoints=False)
    expected = {(x, y): brown for x in range(2, 5) for y in (1, 2)}
    self._check_antialiasing((2, 1.5), (4, 1.5), expected, check_points, set_endpoints=False)
    expected = {(2, 1): brown, (2, 2): FG_GREEN, (2, 3): brown}
    self._check_antialiasing((2, 1.5), (2, 2.5), expected, check_points, set_endpoints=False)
    expected = {(2, 1): brown, (2, 2): greenish}
    self._check_antialiasing((2, 1.5), (2, 1.75), expected, check_points, set_endpoints=False)
    expected = {(x, y): brown for x in (1, 2) for y in range(2, 5)}
    self._check_antialiasing((1.5, 2), (1.5, 4), expected, check_points, set_endpoints=False)
    expected = {(1, 1): brown, (2, 2): FG_GREEN, (3, 3): brown}
    self._check_antialiasing((1.5, 1.5), (2.5, 2.5), expected, check_points, set_endpoints=False)
    expected = {(3, 1): brown, (2, 2): FG_GREEN, (1, 3): brown}
    self._check_antialiasing((2.5, 1.5), (1.5, 2.5), expected, check_points, set_endpoints=False)
    expected = {(2, 1): brown, (2, 2): brown, (3, 2): brown, (3, 3): brown}
    self._check_antialiasing((2, 1.5), (3, 2.5), expected, check_points, set_endpoints=False)
    expected = {(2, 1): greenish, (2, 2): reddish, (3, 2): greenish, (3, 3): reddish, (4, 3): greenish, (4, 4): reddish}
    self._check_antialiasing((2, 1.25), (4, 3.25), expected, check_points, set_endpoints=False)