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
def test_lines__bounding_rect(self):
    """Ensures draw lines returns the correct bounding rect.

        Tests lines with endpoints on and off the surface and a range of
        width/thickness values.
        """
    line_color = pygame.Color('red')
    surf_color = pygame.Color('black')
    width = height = 30
    pos_rect = pygame.Rect((0, 0), (width, height))
    for size in ((width + 5, height + 5), (width - 5, height - 5)):
        surface = pygame.Surface(size, 0, 32)
        surf_rect = surface.get_rect()
        for pos in rect_corners_mids_and_center(surf_rect):
            pos_rect.center = pos
            pts = (pos_rect.midleft, pos_rect.midtop, pos_rect.midright)
            pos = pts[0]
            for thickness in range(-1, 5):
                for closed in (True, False):
                    surface.fill(surf_color)
                    bounding_rect = self.draw_lines(surface, line_color, closed, pts, thickness)
                    if 0 < thickness:
                        expected_rect = create_bounding_rect(surface, surf_color, pos)
                    else:
                        expected_rect = pygame.Rect(pos, (0, 0))
                    self.assertEqual(bounding_rect, expected_rect)