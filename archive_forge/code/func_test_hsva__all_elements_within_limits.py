import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_hsva__all_elements_within_limits(self):
    for c in rgba_combos_Color_generator():
        h, s, v, a = c.hsva
        self.assertTrue(0 <= h <= 360)
        self.assertTrue(0 <= s <= 100)
        self.assertTrue(0 <= v <= 100)
        self.assertTrue(0 <= a <= 100)