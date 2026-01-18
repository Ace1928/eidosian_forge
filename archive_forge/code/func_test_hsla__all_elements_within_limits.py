import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_hsla__all_elements_within_limits(self):
    for c in rgba_combos_Color_generator():
        h, s, l, a = c.hsla
        self.assertTrue(0 <= h <= 360)
        self.assertTrue(0 <= s <= 100)
        self.assertTrue(0 <= l <= 100)
        self.assertTrue(0 <= a <= 100)