from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_mask__negative_size(self):
    """Ensure the mask constructor handles negative sizes correctly."""
    for size in ((1, -1), (-1, 1), (-1, -1)):
        with self.assertRaises(ValueError):
            mask = pygame.Mask(size)