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
def test_arc__kwargs(self):
    """Ensures draw arc accepts the correct kwargs
        with and without a width arg.
        """
    kwargs_list = [{'surface': pygame.Surface((4, 4)), 'color': pygame.Color('yellow'), 'rect': pygame.Rect((0, 0), (3, 2)), 'start_angle': 0.5, 'stop_angle': 3, 'width': 1}, {'surface': pygame.Surface((2, 1)), 'color': (0, 10, 20), 'rect': (0, 0, 2, 2), 'start_angle': 1, 'stop_angle': 3.1}]
    for kwargs in kwargs_list:
        bounds_rect = self.draw_arc(**kwargs)
        self.assertIsInstance(bounds_rect, pygame.Rect)