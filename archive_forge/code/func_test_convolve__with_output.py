from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_convolve__with_output(self):
    """checks that convolution modifies only the correct portion of the output"""
    m = random_mask((10, 10))
    k = pygame.Mask((2, 2))
    k.set_at((0, 0))
    o = pygame.Mask((50, 50))
    test = pygame.Mask((50, 50))
    m.convolve(k, o)
    test.draw(m, (1, 1))
    self.assertIsInstance(o, pygame.mask.Mask)
    assertMaskEqual(self, o, test)
    o.clear()
    test.clear()
    m.convolve(other=k, output=o, offset=Vector2(10, 10))
    test.draw(m, (11, 11))
    self.assertIsInstance(o, pygame.mask.Mask)
    assertMaskEqual(self, o, test)