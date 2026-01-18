from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_centroid__single_column(self):
    """Ensure a mask's centroid is correctly calculated
        when setting points along a single column."""
    width, height = (5, 7)
    mask = pygame.mask.Mask((width, height))
    for x in range(width):
        mask.clear()
        for y in range(height):
            mask.set_at((x, y))
            expected_centroid = (x, y // 2)
            centroid = mask.centroid()
            self.assertEqual(centroid, expected_centroid)