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
def test_anti_aliasing_at_and_outside_the_border(self):
    """Ensures antialiasing works correct at a surface's borders."""
    self.surface = pygame.Surface((10, 10))
    draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
    check_points = [(i, j) for i in range(10) for j in range(10)]
    reddish = (191, 63, 0)
    brown = (127, 127, 0)
    greenish = (63, 191, 0)
    from_point, to_point = ((3, 3), (7, 4))
    should = {(4, 3): greenish, (5, 3): brown, (6, 3): reddish, (4, 4): reddish, (5, 4): brown, (6, 4): greenish}
    for dx, dy in ((-4, 0), (4, 0), (0, -5), (0, -4), (0, -3), (0, 5), (0, 6), (0, 7), (-4, -4), (-4, -3), (-3, -4)):
        first = (from_point[0] + dx, from_point[1] + dy)
        second = (to_point[0] + dx, to_point[1] + dy)
        expected = {(x + dx, y + dy): color for (x, y), color in should.items()}
        self._check_antialiasing(first, second, expected, check_points)