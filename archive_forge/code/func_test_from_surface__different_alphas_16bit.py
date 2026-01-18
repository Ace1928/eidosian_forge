from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_from_surface__different_alphas_16bit(self):
    """Ensures from_surface creates a mask with the correct bits set
        when pixels have different alpha values (16 bit surfaces).

        This test checks the masks created by the from_surface function using
        a 16 bit surface. Each pixel of the surface is set with a different
        alpha value (0-255), but since this is a 16 bit surface the requested
        alpha value can differ from what is actually set. The resulting surface
        will have groups of alpha values which complicates the test as the
        alpha groups will all be set/unset at a given threshold. The setup
        calculates these groups and an expected mask for each. This test data
        is then used to test each alpha grouping over a range of threshold
        values.
        """
    threshold_count = 256
    surface_color = [110, 120, 130, 0]
    expected_size = (threshold_count, 1)
    surface = pygame.Surface(expected_size, SRCALPHA, 16)
    surface.lock()
    for a in range(threshold_count):
        surface_color[3] = a
        surface.set_at((a, 0), surface_color)
    surface.unlock()
    alpha_thresholds = OrderedDict()
    special_thresholds = set()
    for threshold in range(threshold_count):
        alpha = surface.get_at((threshold, 0))[3]
        if alpha not in alpha_thresholds:
            alpha_thresholds[alpha] = [threshold]
        else:
            alpha_thresholds[alpha].append(threshold)
        if threshold < alpha:
            special_thresholds.add(threshold)
    test_data = []
    offset = (0, 0)
    erase_mask = pygame.Mask(expected_size)
    exp_mask = pygame.Mask(expected_size, fill=True)
    for thresholds in alpha_thresholds.values():
        for threshold in thresholds:
            if threshold in special_thresholds:
                test_data.append((threshold, threshold + 1, exp_mask))
            else:
                to_threshold = thresholds[-1] + 1
                for thres in range(to_threshold):
                    erase_mask.set_at((thres, 0), 1)
                exp_mask = pygame.Mask(expected_size, fill=True)
                exp_mask.erase(erase_mask, offset)
                test_data.append((threshold, to_threshold, exp_mask))
                break
    for from_threshold, to_threshold, expected_mask in test_data:
        expected_count = expected_mask.count()
        for threshold in range(from_threshold, to_threshold):
            msg = f'threshold={threshold}'
            mask = pygame.mask.from_surface(surface, threshold)
            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)
            self.assertEqual(mask.count(), expected_count, msg)
            self.assertEqual(mask.overlap_area(expected_mask, offset), expected_count, msg)