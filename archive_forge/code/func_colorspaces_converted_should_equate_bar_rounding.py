import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def colorspaces_converted_should_equate_bar_rounding(self, prop):
    for c in rgba_combos_Color_generator():
        other = pygame.Color(0)
        try:
            setattr(other, prop, getattr(c, prop))
            self.assertTrue(abs(other.r - c.r) <= 1)
            self.assertTrue(abs(other.b - c.b) <= 1)
            self.assertTrue(abs(other.g - c.g) <= 1)
            if not prop in ('cmy', 'i1i2i3'):
                self.assertTrue(abs(other.a - c.a) <= 1)
        except ValueError:
            pass