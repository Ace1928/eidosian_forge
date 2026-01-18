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
def test_ellipse__args_without_width(self):
    """Ensures draw ellipse accepts the args without a width."""
    bounds_rect = self.draw_ellipse(pygame.Surface((2, 2)), (1, 1, 1, 99), pygame.Rect((1, 1), (1, 1)))
    self.assertIsInstance(bounds_rect, pygame.Rect)