import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__int_arg(self):
    """Ensures Color objects can be created using one int value."""
    for value in (0, 4294967295, 2864434397):
        color = pygame.Color(value)
        self.assertEqual(color.r, value >> 24 & 255)
        self.assertEqual(color.g, value >> 16 & 255)
        self.assertEqual(color.b, value >> 8 & 255)
        self.assertEqual(color.a, value & 255)