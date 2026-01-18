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
def test_circle_negative_radius(self):
    """Ensures negative radius circles return zero sized bounding rect."""
    surf = pygame.Surface((200, 200))
    color = (0, 0, 0, 50)
    center = (surf.get_height() // 2, surf.get_height() // 2)
    bounding_rect = self.draw_circle(surf, color, center, radius=-1, width=1)
    self.assertEqual(bounding_rect.size, (0, 0))