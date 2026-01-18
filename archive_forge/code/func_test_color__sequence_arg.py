import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__sequence_arg(self):
    """Ensures Color objects can be created using tuples/lists."""
    color_values = (33, 44, 55, 66)
    for seq_type in (tuple, list):
        color = pygame.Color(seq_type(color_values))
        self.assertEqual(color.r, color_values[0])
        self.assertEqual(color.g, color_values[1])
        self.assertEqual(color.b, color_values[2])
        self.assertEqual(color.a, color_values[3])