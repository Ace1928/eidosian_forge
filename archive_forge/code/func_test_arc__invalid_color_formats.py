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
def test_arc__invalid_color_formats(self):
    """Ensures draw arc handles invalid color formats correctly."""
    pos = (1, 1)
    surface = pygame.Surface((4, 3))
    kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (2, 2)), 'start_angle': 5, 'stop_angle': 6.1, 'width': 1}
    for expected_color in (2.3, self):
        kwargs['color'] = expected_color
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(**kwargs)