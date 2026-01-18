from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_copy__empty(self):
    """Ensures copy works correctly on an empty mask."""
    for width in (31, 32, 33, 63, 64, 65):
        for height in (31, 32, 33, 63, 64, 65):
            mask = pygame.mask.Mask((width, height))
            for mask_copy in (mask.copy(), copy.copy(mask)):
                self.assertIsInstance(mask_copy, pygame.mask.Mask)
                self.assertIsNot(mask_copy, mask)
                assertMaskEqual(self, mask_copy, mask)