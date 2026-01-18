from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
def test_fill__bit_boundaries(self):
    """Ensures masks of different sizes are filled correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
    for height in range(1, 4):
        for width in range(1, 66):
            mask = pygame.mask.Mask((width, height))
            expected_count = width * height
            mask.fill()
            self.assertEqual(mask.count(), expected_count, f'size=({width}, {height})')