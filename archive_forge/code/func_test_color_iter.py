import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color_iter(self):
    c = pygame.Color(50, 100, 150, 200)
    color_iterator = c.__iter__()
    for i, val in enumerate(color_iterator):
        self.assertEqual(c[i], val)