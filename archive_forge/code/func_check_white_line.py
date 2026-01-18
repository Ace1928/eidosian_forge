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
def check_white_line(start, end):
    surf.fill((0, 0, 0))
    pygame.draw.line(surf, (255, 255, 255), start, end, 30)
    BLACK = (0, 0, 0, 255)
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if surf.get_at((x, y)) == BLACK:
                self.assertTrue(white_surrounded_pixels(x, y) < 3)