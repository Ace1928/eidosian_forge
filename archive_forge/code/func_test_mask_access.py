from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_mask_access(self):
    """do the set_at, and get_at parts work correctly?"""
    m = pygame.Mask((10, 10))
    m.set_at((0, 0), 1)
    self.assertEqual(m.get_at((0, 0)), 1)
    m.set_at((9, 0), 1)
    self.assertEqual(m.get_at((9, 0)), 1)
    self.assertRaises(IndexError, lambda: m.get_at((-1, 0)))
    self.assertRaises(IndexError, lambda: m.set_at((-1, 0), 1))
    self.assertRaises(IndexError, lambda: m.set_at((10, 0), 1))
    self.assertRaises(IndexError, lambda: m.set_at((0, 10), 1))