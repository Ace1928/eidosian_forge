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
def test_line__color(self):
    """Tests if the line drawn is the correct color."""
    pos = (0, 0)
    for surface in self._create_surfaces():
        for expected_color in self.COLORS:
            self.draw_line(surface, expected_color, pos, (1, 0))
            self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')