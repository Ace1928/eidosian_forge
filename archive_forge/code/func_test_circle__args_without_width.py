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
def test_circle__args_without_width(self):
    """Ensures draw circle accepts the args without a width and
        quadrants."""
    bounds_rect = self.draw_circle(pygame.Surface((2, 2)), (0, 0, 0, 50), (1, 1), 1)
    self.assertIsInstance(bounds_rect, pygame.Rect)