from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_get_fonts(self):
    fnts = pygame_font.get_fonts()
    self.assertTrue(fnts, msg=repr(fnts))
    for name in fnts:
        self.assertTrue(isinstance(name, str), name)
        self.assertFalse(any((c.isupper() for c in name)))
        self.assertTrue(name.isalnum(), name)