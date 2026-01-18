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
def test_ellipse__max_width(self):
    """Ensures an ellipse with max width (and greater) is drawn correctly."""
    ellipse_color = pygame.Color('yellow')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((40, 40))
    rect = pygame.Rect((0, 0), (31, 21))
    rect.center = surface.get_rect().center
    max_thickness = (min(*rect.size) + 1) // 2
    for thickness in range(max_thickness, max_thickness + 3):
        surface.fill(surface_color)
        self.draw_ellipse(surface, ellipse_color, rect, thickness)
        surface.lock()
        for y in range(rect.top, rect.bottom):
            self.assertEqual(surface.get_at((rect.centerx, y)), ellipse_color)
        for x in range(rect.left, rect.right):
            self.assertEqual(surface.get_at((x, rect.centery)), ellipse_color)
        self.assertEqual(surface.get_at((rect.centerx, rect.top - 1)), surface_color)
        self.assertEqual(surface.get_at((rect.centerx, rect.bottom + 1)), surface_color)
        self.assertEqual(surface.get_at((rect.left - 1, rect.centery)), surface_color)
        self.assertEqual(surface.get_at((rect.right + 1, rect.centery)), surface_color)
        surface.unlock()