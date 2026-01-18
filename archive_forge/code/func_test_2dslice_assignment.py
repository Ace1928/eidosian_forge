import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_2dslice_assignment(self):
    w = 2 * 5 * 8
    h = 3 * 5 * 9
    sf = pygame.Surface((w, h), 0, 32)
    ar = pygame.PixelArray(sf)
    size = (w, h)
    strides = (1, w)
    offset = 0
    self._test_assignment(sf, ar, size, strides, offset)
    xslice = slice(None, None, 2)
    yslice = slice(None, None, 3)
    ar, size, strides, offset = self._array_slice(ar, size, (xslice, yslice), strides, offset)
    self._test_assignment(sf, ar, size, strides, offset)
    xslice = slice(5, None, 5)
    yslice = slice(5, None, 5)
    ar, size, strides, offset = self._array_slice(ar, size, (xslice, yslice), strides, offset)
    self._test_assignment(sf, ar, size, strides, offset)