import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_i1i2i3__all_elements_within_limits(self):
    for c in rgba_combos_Color_generator():
        i1, i2, i3 = c.i1i2i3
        self.assertTrue(0 <= i1 <= 1)
        self.assertTrue(-0.5 <= i2 <= 0.5)
        self.assertTrue(-0.5 <= i3 <= 0.5)