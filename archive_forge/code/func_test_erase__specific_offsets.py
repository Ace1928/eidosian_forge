from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_erase__specific_offsets(self):
    """Ensure an offset mask can erase another mask.

        Testing the specific case of:
            -both masks are wider than 32 bits
            -a positive offset is used
            -the mask calling erase() is wider than the mask passed in
        """
    mask1 = pygame.mask.Mask((65, 5))
    mask2 = pygame.mask.Mask((33, 3), fill=True)
    expected_mask = pygame.Mask(mask1.get_size())
    rect1 = mask1.get_rect()
    rect2 = mask2.get_rect()
    corner_rect = rect1.inflate(-2, -2)
    for corner in ('topleft', 'topright', 'bottomright', 'bottomleft'):
        setattr(rect2, corner, getattr(corner_rect, corner))
        offset = rect2.topleft
        msg = f'offset={offset}'
        overlap_rect = rect1.clip(rect2)
        expected_mask.fill()
        for x in range(overlap_rect.left, overlap_rect.right):
            for y in range(overlap_rect.top, overlap_rect.bottom):
                expected_mask.set_at((x, y), 0)
        mask1.fill()
        mask1.erase(mask2, Vector2(offset))
        assertMaskEqual(self, mask1, expected_mask, msg)