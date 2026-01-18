from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_connected_components__negative_min_with_full_mask(self):
    """Ensures connected_components() properly handles negative min values
        when the mask is full.

        Negative and zero values for the min parameter (minimum number of bits
        per connected component) equate to setting it to one.
        """
    mask_size = (64, 11)
    mask = pygame.mask.Mask(mask_size, fill=True)
    mask_count = mask.count()
    expected_len = 1
    connected_comps = mask.connected_components(-2)
    self.assertEqual(len(connected_comps), expected_len)
    assertMaskEqual(self, connected_comps[0], mask)
    self.assertEqual(mask.count(), mask_count)
    self.assertEqual(mask.get_size(), mask_size)