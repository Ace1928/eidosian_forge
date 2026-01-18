import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_color__color_object_arg(self):
    """Ensures Color objects can be created using Color objects."""
    color_args = (10, 20, 30, 40)
    color_obj = pygame.Color(*color_args)
    new_color_obj = pygame.Color(color_obj)
    self.assertIsInstance(new_color_obj, pygame.Color)
    self.assertEqual(new_color_obj, color_obj)
    self.assertEqual(new_color_obj.r, color_args[0])
    self.assertEqual(new_color_obj.g, color_args[1])
    self.assertEqual(new_color_obj.b, color_args[2])
    self.assertEqual(new_color_obj.a, color_args[3])