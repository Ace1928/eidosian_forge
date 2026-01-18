from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_count__full_mask(self):
    """Ensure a full mask's set bits are correctly counted."""
    width, height = (17, 97)
    expected_size = (width, height)
    expected_count = width * height
    mask = pygame.mask.Mask(expected_size, fill=True)
    count = mask.count()
    self.assertEqual(count, expected_count)
    self.assertEqual(mask.get_size(), expected_size)