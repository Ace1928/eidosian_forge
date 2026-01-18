from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_centroid__all_corners(self):
    """Ensure a mask's centroid is correctly calculated
        when its corners are set."""
    mask = pygame.mask.Mask((5, 7))
    expected_centroid = mask.get_rect().center
    for corner in corners(mask):
        mask.set_at(corner)
    centroid = mask.centroid()
    self.assertEqual(centroid, expected_centroid)