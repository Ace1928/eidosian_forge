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
def test_arc(self):
    """Ensure draw arc works correctly."""
    black = pygame.Color('black')
    red = pygame.Color('red')
    surface = pygame.Surface((100, 150))
    surface.fill(black)
    rect = (0, 0, 80, 40)
    start_angle = 0.0
    stop_angle = 3.14
    width = 3
    pygame.draw.arc(surface, red, rect, start_angle, stop_angle, width)
    pygame.image.save(surface, 'arc.png')
    x = 20
    for y in range(2, 5):
        self.assertEqual(surface.get_at((x, y)), red)
    self.assertEqual(surface.get_at((0, 0)), black)