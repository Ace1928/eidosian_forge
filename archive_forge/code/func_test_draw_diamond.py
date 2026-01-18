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
def test_draw_diamond(self):
    pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
    self.draw_polygon(self.surface, GREEN, DIAMOND, 0)
    for x, y in DIAMOND:
        self.assertEqual(self.surface.get_at((x, y)), GREEN, msg=str((x, y)))
    for x in range(2, 5):
        for y in range(2, 5):
            self.assertEqual(self.surface.get_at((x, y)), GREEN)