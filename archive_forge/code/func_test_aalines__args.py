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
def test_aalines__args(self):
    """Ensures draw aalines accepts the correct args."""
    bounds_rect = self.draw_aalines(pygame.Surface((3, 3)), (0, 10, 0, 50), False, ((0, 0), (1, 1)), 1)
    self.assertIsInstance(bounds_rect, pygame.Rect)