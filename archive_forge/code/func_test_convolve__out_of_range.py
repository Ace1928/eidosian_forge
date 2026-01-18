from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_convolve__out_of_range(self):
    full = pygame.Mask((2, 2), fill=True)
    pts_data = (((0, 3), 0), ((0, 2), 3), ((-2, -2), 1), ((-3, -3), 0))
    for pt, expected_count in pts_data:
        convolve_mask = full.convolve(full, None, pt)
        self.assertIsInstance(convolve_mask, pygame.mask.Mask)
        self.assertEqual(convolve_mask.count(), expected_count)