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
def test_ellipse__args_with_width_gt_radius(self):
    """Ensures draw ellipse accepts the args with
        width > rect.w // 2 and width > rect.h // 2.
        """
    rect = pygame.Rect((0, 0), (4, 4))
    bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), rect, rect.w // 2 + 1)
    self.assertIsInstance(bounds_rect, pygame.Rect)
    bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), rect, rect.h // 2 + 1)
    self.assertIsInstance(bounds_rect, pygame.Rect)