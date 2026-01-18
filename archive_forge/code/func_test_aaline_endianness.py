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
def test_aaline_endianness(self):
    """test color component order"""
    for depth in (24, 32):
        surface = pygame.Surface((5, 3), 0, depth)
        surface.fill(pygame.Color(0, 0, 0))
        self.draw_aaline(surface, pygame.Color(255, 0, 0), (0, 1), (2, 1), 1)
        self.assertGreater(surface.get_at((1, 1)).r, 0, 'there should be red here')
        surface.fill(pygame.Color(0, 0, 0))
        self.draw_aaline(surface, pygame.Color(0, 0, 255), (0, 1), (2, 1), 1)
        self.assertGreater(surface.get_at((1, 1)).b, 0, 'there should be blue here')