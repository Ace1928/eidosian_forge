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
def test_aaline__args_without_blend(self):
    """Ensures draw aaline accepts the args without a blend."""
    bounds_rect = self.draw_aaline(pygame.Surface((2, 2)), (0, 0, 0, 50), (0, 0), (2, 2))
    self.assertIsInstance(bounds_rect, pygame.Rect)