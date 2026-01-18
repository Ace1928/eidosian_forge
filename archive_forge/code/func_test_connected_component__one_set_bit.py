from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_connected_component__one_set_bit(self):
    """Ensure a mask's connected component is correctly calculated
        when the coordinate's bit is set with a connected component of 1 bit.
        """
    width, height = (71, 67)
    expected_size = (width, height)
    original_mask = pygame.mask.Mask(expected_size, fill=True)
    xset, yset = (width // 2, height // 2)
    set_pos = (xset, yset)
    expected_offset = (xset - 1, yset - 1)
    expected_pattern = self._draw_component_pattern_box(original_mask, 3, expected_offset, inverse=True)
    expected_count = 1
    original_count = original_mask.count()
    mask = original_mask.connected_component(set_pos)
    self.assertIsInstance(mask, pygame.mask.Mask)
    self.assertEqual(mask.count(), expected_count)
    self.assertEqual(mask.get_size(), expected_size)
    self.assertEqual(mask.overlap_area(expected_pattern, expected_offset), expected_count)
    self.assertEqual(original_mask.count(), original_count)
    self.assertEqual(original_mask.get_size(), expected_size)
    self.assertEqual(original_mask.overlap_area(expected_pattern, expected_offset), expected_count)