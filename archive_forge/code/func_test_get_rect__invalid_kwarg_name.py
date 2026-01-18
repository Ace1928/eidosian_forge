from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
def test_get_rect__invalid_kwarg_name(self):
    """Ensures get_rect detects invalid kwargs."""
    mask = pygame.mask.Mask((1, 2))
    with self.assertRaises(AttributeError):
        rect = mask.get_rect(righte=11)
    with self.assertRaises(AttributeError):
        rect = mask.get_rect(toplef=(1, 1))
    with self.assertRaises(AttributeError):
        rect = mask.get_rect(move=(3, 2))