import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__sequence_arg_invalid_value_without_alpha(self):
    """Ensures invalid sequences are detected when creating Color objects
        without providing an alpha.
        """
    cls = pygame.Color
    for seq_type in (tuple, list):
        self.assertRaises(ValueError, cls, seq_type((256, 90, 80)))
        self.assertRaises(ValueError, cls, seq_type((100, 256, 80)))
        self.assertRaises(ValueError, cls, seq_type((100, 90, 256)))