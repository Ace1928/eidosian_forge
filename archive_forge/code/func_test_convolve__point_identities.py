from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_convolve__point_identities(self):
    """Convolving with a single point is the identity, while convolving a point with something flips it."""
    m = random_mask((100, 100))
    k = pygame.Mask((1, 1))
    k.set_at((0, 0))
    convolve_mask = m.convolve(k)
    self.assertIsInstance(convolve_mask, pygame.mask.Mask)
    assertMaskEqual(self, m, convolve_mask)
    convolve_mask = k.convolve(k.convolve(m))
    self.assertIsInstance(convolve_mask, pygame.mask.Mask)
    assertMaskEqual(self, m, convolve_mask)