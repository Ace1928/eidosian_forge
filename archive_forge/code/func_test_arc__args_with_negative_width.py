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
def test_arc__args_with_negative_width(self):
    """Ensures draw arc accepts the args with negative width."""
    bounds_rect = self.draw_arc(pygame.Surface((3, 3)), (10, 10, 50, 50), (1, 1, 2, 2), 0, 1, -1)
    self.assertIsInstance(bounds_rect, pygame.Rect)
    self.assertEqual(bounds_rect, pygame.Rect(1, 1, 0, 0))