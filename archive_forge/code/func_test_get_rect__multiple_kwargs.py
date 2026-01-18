from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_get_rect__multiple_kwargs(self):
    """Ensures get_rect supports multiple rect attribute kwargs."""
    mask = pygame.mask.Mask((5, 4))
    expected_rect = pygame.Rect((0, 0), (0, 0))
    kwargs = {'x': 7.1, 'top': -1, 'size': Vector2(2, 3.2)}
    for attrib, value in kwargs.items():
        setattr(expected_rect, attrib, value)
    rect = mask.get_rect(**kwargs)
    self.assertEqual(rect, expected_rect)