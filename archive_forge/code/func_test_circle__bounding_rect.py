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
def test_circle__bounding_rect(self):
    """Ensures draw circle returns the correct bounding rect.

        Tests circles on and off the surface and a range of width/thickness
        values.
        """
    circle_color = pygame.Color('red')
    surf_color = pygame.Color('black')
    max_radius = 3
    surface = pygame.Surface((30, 30), 0, 32)
    surf_rect = surface.get_rect()
    big_rect = surf_rect.inflate(max_radius * 2 - 1, max_radius * 2 - 1)
    for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
        for radius in range(max_radius + 1):
            for thickness in range(radius + 1):
                surface.fill(surf_color)
                bounding_rect = self.draw_circle(surface, circle_color, pos, radius, thickness)
                expected_rect = create_bounding_rect(surface, surf_color, pos)
                with self.subTest(surface=surface, circle_color=circle_color, pos=pos, radius=radius, thickness=thickness):
                    self.assertEqual(bounding_rect, expected_rect)