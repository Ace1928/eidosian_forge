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
def test_line__color_with_thickness(self):
    """Ensures a thick line is drawn using the correct color."""
    from_x = 5
    to_x = 10
    y = 5
    for surface in self._create_surfaces():
        for expected_color in self.COLORS:
            self.draw_line(surface, expected_color, (from_x, y), (to_x, y), 5)
            for pos in ((x, y + i) for i in (-2, 0, 2) for x in (from_x, to_x)):
                self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')