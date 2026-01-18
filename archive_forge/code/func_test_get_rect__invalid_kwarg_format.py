from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_get_rect__invalid_kwarg_format(self):
    """Ensures get_rect detects invalid kwarg formats."""
    mask = pygame.mask.Mask((3, 11))
    with self.assertRaises(TypeError):
        rect = mask.get_rect(right='1')
    with self.assertRaises(TypeError):
        rect = mask.get_rect(bottom=(1,))
    with self.assertRaises(TypeError):
        rect = mask.get_rect(centerx=(1, 1))
    with self.assertRaises(TypeError):
        rect = mask.get_rect(midleft=(1, '1'))
    with self.assertRaises(TypeError):
        rect = mask.get_rect(topright=(1,))
    with self.assertRaises(TypeError):
        rect = mask.get_rect(bottomleft=(1, 2, 3))
    with self.assertRaises(TypeError):
        rect = mask.get_rect(midbottom=1)