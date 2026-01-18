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
def test_ellipse__args(self):
    """Ensures draw ellipse accepts the correct args."""
    bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), pygame.Rect((0, 0), (3, 2)), 1)
    self.assertIsInstance(bounds_rect, pygame.Rect)