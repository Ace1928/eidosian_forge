import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__html_str_arg(self):
    """Ensures Color objects can be created using html strings."""
    color = pygame.Color('#a1B2c3D4')
    self.assertEqual(color.r, 161)
    self.assertEqual(color.g, 178)
    self.assertEqual(color.b, 195)
    self.assertEqual(color.a, 212)