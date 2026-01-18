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
def test_circle__diameter(self):
    """Ensures draw circle is twice size of radius high and wide."""
    surf = pygame.Surface((200, 200))
    color = (0, 0, 0, 50)
    center = (surf.get_height() // 2, surf.get_height() // 2)
    width = 1
    radius = 6
    for radius in range(1, 65):
        bounding_rect = self.draw_circle(surf, color, center, radius, width)
        self.assertEqual(bounding_rect.width, radius * 2)
        self.assertEqual(bounding_rect.height, radius * 2)