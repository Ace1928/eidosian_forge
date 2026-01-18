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
def test_aaline__gaps(self):
    """Tests if the aaline drawn contains any gaps.

        See: #512
        """
    expected_color = (255, 255, 255)
    for surface in self._create_surfaces():
        width = surface.get_width()
        self.draw_aaline(surface, expected_color, (0, 0), (width - 1, 0))
        for x in range(width):
            pos = (x, 0)
            self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')