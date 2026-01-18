import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def colorspaces_converted_should_not_raise(self, prop):
    fails = 0
    x = 0
    for c in rgba_combos_Color_generator():
        x += 1
        other = pygame.Color(0)
        try:
            setattr(other, prop, getattr(c, prop))
        except ValueError:
            fails += 1
    self.assertTrue(x > 0, 'x is combination counter, 0 means no tests!')
    self.assertTrue((fails, x) == (0, x))