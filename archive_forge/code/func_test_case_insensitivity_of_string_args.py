import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_case_insensitivity_of_string_args(self):
    self.assertEqual(pygame.color.Color('red'), pygame.color.Color('Red'))