import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color_contains(self):
    c = pygame.Color(50, 60, 70)
    self.assertTrue(c.__contains__(50))
    self.assertTrue(60 in c)
    self.assertTrue(70 in c)
    self.assertFalse(100 in c)
    self.assertFalse(c.__contains__(10))
    self.assertRaises(TypeError, lambda: 'string' in c)
    self.assertRaises(TypeError, lambda: 3.14159 in c)