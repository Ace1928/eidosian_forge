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
def test_ellipse__kwargs_order_independent(self):
    """Ensures draw ellipse's kwargs are not order dependent."""
    bounds_rect = self.draw_ellipse(color=(1, 2, 3), surface=pygame.Surface((3, 2)), width=0, rect=pygame.Rect((1, 0), (1, 1)))
    self.assertIsInstance(bounds_rect, pygame.Rect)