import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_assign_size_mismatch(self):
    sf = pygame.Surface((7, 11), 0, 32)
    ar = pygame.PixelArray(sf)
    self.assertRaises(ValueError, ar.__setitem__, Ellipsis, ar[:, 0:2])
    self.assertRaises(ValueError, ar.__setitem__, Ellipsis, ar[0:2, :])