from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_mask__fill_kwarg(self):
    """Ensure masks are created correctly using the fill keyword."""
    width, height = (37, 47)
    expected_size = (width, height)
    fill_counts = {True: width * height, False: 0}
    for fill, expected_count in fill_counts.items():
        msg = f'fill={fill}'
        mask = pygame.mask.Mask(expected_size, fill=fill)
        self.assertIsInstance(mask, pygame.mask.Mask, msg)
        self.assertEqual(mask.count(), expected_count, msg)
        self.assertEqual(mask.get_size(), expected_size, msg)