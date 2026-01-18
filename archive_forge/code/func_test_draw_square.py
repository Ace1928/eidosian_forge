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
def test_draw_square(self):
    self.draw_polygon(self.surface, RED, SQUARE, 0)
    for x in range(4):
        for y in range(4):
            self.assertEqual(self.surface.get_at((x, y)), RED)