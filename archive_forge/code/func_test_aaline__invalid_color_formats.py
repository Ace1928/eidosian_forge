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
def test_aaline__invalid_color_formats(self):
    """Ensures draw aaline handles invalid color formats correctly."""
    kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'start_pos': (1, 1), 'end_pos': (2, 1)}
    for expected_color in (2.3, self):
        kwargs['color'] = expected_color
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(**kwargs)