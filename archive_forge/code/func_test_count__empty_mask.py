from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_count__empty_mask(self):
    """Ensure an empty mask's set bits are correctly counted."""
    expected_count = 0
    expected_size = (13, 27)
    mask = pygame.mask.Mask(expected_size)
    count = mask.count()
    self.assertEqual(count, expected_count)
    self.assertEqual(mask.get_size(), expected_size)