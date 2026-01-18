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
def test_color_validation(self):
    surf = pygame.Surface((10, 10))
    colors = (123456, (1, 10, 100), RED, '#ab12df', 'red')
    points = ((0, 0), (1, 1), (1, 0))
    for col in colors:
        draw.line(surf, col, (0, 0), (1, 1))
        draw.aaline(surf, col, (0, 0), (1, 1))
        draw.aalines(surf, col, True, points)
        draw.lines(surf, col, True, points)
        draw.arc(surf, col, pygame.Rect(0, 0, 3, 3), 15, 150)
        draw.ellipse(surf, col, pygame.Rect(0, 0, 3, 6), 1)
        draw.circle(surf, col, (7, 3), 2)
        draw.polygon(surf, col, points, 0)
    for col in (1.256, object(), None):
        with self.assertRaises(TypeError):
            draw.line(surf, col, (0, 0), (1, 1))
        with self.assertRaises(TypeError):
            draw.aaline(surf, col, (0, 0), (1, 1))
        with self.assertRaises(TypeError):
            draw.aalines(surf, col, True, points)
        with self.assertRaises(TypeError):
            draw.lines(surf, col, True, points)
        with self.assertRaises(TypeError):
            draw.arc(surf, col, pygame.Rect(0, 0, 3, 3), 15, 150)
        with self.assertRaises(TypeError):
            draw.ellipse(surf, col, pygame.Rect(0, 0, 3, 6), 1)
        with self.assertRaises(TypeError):
            draw.circle(surf, col, (7, 3), 2)
        with self.assertRaises(TypeError):
            draw.polygon(surf, col, points, 0)