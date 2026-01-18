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
def test_lines__gaps(self):
    """Tests if the lines drawn contain any gaps.

        Draws lines around the border of the given surface and checks if
        all borders of the surface contain any gaps.
        """
    expected_color = (255, 255, 255)
    for surface in self._create_surfaces():
        self.draw_lines(surface, expected_color, True, corners(surface))
        for pos, color in border_pos_and_color(surface):
            self.assertEqual(color, expected_color, f'pos={pos}')