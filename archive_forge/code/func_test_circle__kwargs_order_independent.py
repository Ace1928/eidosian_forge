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
def test_circle__kwargs_order_independent(self):
    """Ensures draw circle's kwargs are not order dependent."""
    bounds_rect = self.draw_circle(draw_top_right=False, color=(10, 20, 30), surface=pygame.Surface((3, 2)), width=0, draw_bottom_left=False, center=(1, 0), draw_bottom_right=False, radius=2, draw_top_left=True)
    self.assertIsInstance(bounds_rect, pygame.Rect)