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
def test_ellipse__no_holes(self):
    width = 80
    height = 70
    surface = pygame.Surface((width + 1, height))
    rect = pygame.Rect(0, 0, width, height)
    for thickness in range(1, 37, 5):
        surface.fill('BLACK')
        self.draw_ellipse(surface, 'RED', rect, thickness)
        for y in range(height):
            number_of_changes = 0
            drawn_pixel = False
            for x in range(width + 1):
                if not drawn_pixel and surface.get_at((x, y)) == pygame.Color('RED') or (drawn_pixel and surface.get_at((x, y)) == pygame.Color('BLACK')):
                    drawn_pixel = not drawn_pixel
                    number_of_changes += 1
            if y < thickness or y > height - thickness - 1:
                self.assertEqual(number_of_changes, 2)
            else:
                self.assertEqual(number_of_changes, 4)