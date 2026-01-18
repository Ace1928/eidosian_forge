from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_get_at__out_of_bounds(self):
    """Ensure get_at() checks bounds."""
    width, height = (11, 3)
    mask = pygame.mask.Mask((width, height))
    with self.assertRaises(IndexError):
        mask.get_at((width, 0))
    with self.assertRaises(IndexError):
        mask.get_at((0, height))
    with self.assertRaises(IndexError):
        mask.get_at((-1, 0))
    with self.assertRaises(IndexError):
        mask.get_at((0, -1))